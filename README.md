# GHOST-VARIANCE VERLET OPTIMIZER

## How to use:

In your training loop, simply create the optimizer, compute the ghost gradients, and tell the optimizer to step.

```python
# Training loop
#...
from GVVOptimizer import GhostVarianceVerlet, compute_ghost_and_mean_gradients

optimizer = GhostVarianceVerlet(model.parameters(), lr=1e-2, mu_max=0.9, weight_decay=1e-4, compress=True)
ghost_grads = compute_ghost_gradients(model, inputs, targets, loss_fn, K=K_ghosts)
optimizer.step(ghost_grads=ghost_grads)
#...
```