import torch
from torch.optim import Optimizer


def quantize_blockwise_int8(tensor, block_size=64):
    """
    Compresses a float tensor into blockwise scaled signed 8-bit integers.
    Saves ~75% of memory per tensor relative to FP32.
    """
    ori_shape = tensor.shape
    flat = tensor.view(-1)
    n = flat.numel()

    # Pad to make flat tensor a multiple of block_size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device, dtype=flat.dtype)])

    # Reshape into blocks
    blocked = flat.view(-1, block_size)

    # Find absolute scales per block
    scales = torch.max(torch.abs(blocked), dim=1, keepdim=True)[0]
    scales = torch.clamp(scales, min=1e-12) # Avoid division by zero

    # Quantize to int8 range [-127, 127]
    quantized = torch.clamp(torch.round((blocked / scales) * 127.0), -127, 127).to(torch.int8)

    return quantized, scales.half(), ori_shape, pad_len  # Store scales in FP16 to save extra space


def dequantize_blockwise_int8(quantized, scales, ori_shape, pad_len):
    """
    Decompresses the blockwise INT8 representation back to its float format.
    """
    # Cast to float, scale back, and flatten
    blocked = quantized.to(scales.dtype) * (scales / 127.0)
    flat = blocked.view(-1)

    if pad_len > 0:
        flat = flat[:-pad_len]

    return flat.view(ori_shape).to(torch.float32)


def compute_ghost_gradients(model, inputs, targets, loss_fn, K):
    """
    Splits the forward & backward passes into K distinct microbatches
    to extract microbatch gradients for GVV, but populates p.grad
    with the average gradient to maintain compatibility with training metrics.
    """
    input_chunks = torch.chunk(inputs, K, dim=0)
    target_chunks = torch.chunk(targets, K, dim=0)

    ghost_grads = {p: [] for p in model.parameters() if p.requires_grad}

    for i in range(K):
        model.zero_grad()
        outputs = model(input_chunks[i])
        loss = loss_fn(outputs, target_chunks[i])
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                if p.grad is not None:
                    ghost_grads[p].append(p.grad.detach().clone())
                else:
                    ghost_grads[p].append(torch.zeros_like(p))

    # Clean up and reset parameter gradients to the correct global batch mean
    model.zero_grad()
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.stack(ghost_grads[p], dim=0).mean(dim=0)

    return ghost_grads


class GhostVarianceVerlet(Optimizer):
    """
    Ghost-Variance Verlet (GVV)

    An adaptive, memory-efficient optimizer that uses intra-batch microbatch
    variance to replace AdamW's persistent 1st & 2nd moment EMAs. It relies
    on one single compressed displacement buffer (Delta) to store dynamics.
    """
    def __init__(self, params, lr=1e-3, weight_decay=1e-2,
                 eps_abs=1e-8, eps_rel=1e-3, mu_max=0.9,
                 block_size=64, use_compression=True):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        eps_abs=eps_abs, eps_rel=eps_rel, mu_max=mu_max,
                        block_size=block_size, use_compression=use_compression)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, ghost_grads, closure=None):
        """
        Performs a single optimization step.
        ghost_grads: dict mapping parameters to a list of K microbatch gradient tensors.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps_abs = group['eps_abs']
            eps_rel = group['eps_rel']
            mu_max = group['mu_max']
            block_size = group['block_size']
            use_compression = group['use_compression']

            for p in group['params']:
                if p.grad is None or p not in ghost_grads:
                    continue

                grads = ghost_grads[p]
                K = len(grads)
                if K < 2:
                    raise ValueError("GVV requires K >= 2 ghost batches to estimate variance.")

                # Stack microbatch gradients to compute mean and variance
                grads_stack = torch.stack(grads, dim=0) # [K, *param_shape]

                # Mean gradient over the batch (G)
                G = grads_stack.mean(dim=0)

                # Unbiased variance of microbatches (V)
                V = grads_stack.var(dim=0, unbiased=True)

                # Variance of the batch mean estimator (N = V / K)
                N = V / K

                # Adaptive stabilizing floor: eps_l = eps_abs + eps_rel * RMS(G)
                rms_G = torch.sqrt(torch.mean(G ** 2))
                eps_l = eps_abs + eps_rel * rms_G

                # Compute Ghost-SNR Force: F = G / sqrt(G^2 + N + eps_l^2)
                denom = torch.sqrt(G**2 + N + eps_l**2)
                F = G / denom

                # Layerwise reliability index: R = mean(G^2) / mean(G^2 + N + eps_l^2)
                num_R = torch.mean(G**2)
                den_R = torch.mean(G**2 + N + eps_l**2)
                R = torch.clamp(num_R / (den_R + 1e-12), 0.0, 1.0)

                # Gate inertia by current reliability score
                mu_t = mu_max * R

                # Fetch or initialize displacement state (Delta)
                state = self.state[p]
                if len(state) == 0:
                    if use_compression:
                        quantized, scales, ori_shape, pad_len = quantize_blockwise_int8(
                            torch.zeros_like(p), block_size
                        )
                        state['quantized_Delta'] = quantized
                        state['scales'] = scales
                        state['ori_shape'] = ori_shape
                        state['pad_len'] = pad_len
                    else:
                        state['Delta'] = torch.zeros_like(p)

                # Fetch Delta (decompressing if needed)
                if use_compression:
                    Delta = dequantize_blockwise_int8(
                        state['quantized_Delta'],
                        state['scales'],
                        state['ori_shape'],
                        state['pad_len']
                    ).to(p.device)
                else:
                    Delta = state['Delta']

                # Update formulation: U_t = mu_t * Delta_t - lr * F_t
                U = mu_t * Delta - lr * F

                # Apply decoupled weight decay directly to weights
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # Apply update displacement to parameter
                p.add_(U)

                # Store back the updated displacement
                if use_compression:
                    quantized, scales, ori_shape, pad_len = quantize_blockwise_int8(U, block_size)
                    state['quantized_Delta'] = quantized
                    state['scales'] = scales
                    state['ori_shape'] = ori_shape
                    state['pad_len'] = pad_len
                else:
                    state['Delta'].copy_(U)

        return loss