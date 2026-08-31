# TVRdiff-Ultra

Matrix-free total-variation regularized differentiation for 1D signals, implemented in JAX.

Given a noisy 1D signal, `TVRdiff_Ultra` recovers a piecewise-smooth derivative estimate by minimizing a total-variation regularized least-squares objective. The signal is split into non-overlapping sub-sequences, each solved independently with Newton's method, where every Newton step is solved via matrix-free conjugate gradient (no dense or sparse `n x n` matrices are ever formed). Sub-sequences are processed in parallel across the batch dimension using `vmap`, making the method efficient on both CPU and GPU/TPU.

## Requirements

- `jax`
- `numpy`

## Installation

Drop `TVRdiff_Ultra` into your project — no packaging required.

```bash
pip install jax numpy
```

## Usage

```python
import numpy as np
from tvrdiff_ultra import TVRdiff_Ultra

# Noisy signal
t = np.linspace(0, 10, 2000)
data = np.sin(t) + 0.05 * np.random.randn(len(t))

result = TVRdiff_Ultra(
    data,
    n=200,        # sub-sequence length
    h=t[1] - t[0],  # sample spacing
    alpha=0.05,   # regularization strength
)

denoised = result["denoised_data"]
derivative = result["diff_data"]
```

## Parameters

| Parameter | Description |
|---|---|
| `data` | 1D input signal to differentiate. |
| `n` | Length of each sub-sequence the signal is split into. |
| `h` | Sample spacing between consecutive points in `data`. |
| `alpha` | Regularization strength; higher values yield smoother, more piecewise-constant derivatives. |
| `eps` | Smoothing constant in the TV term to keep it differentiable at zero-slope points (default `1e-6`). |
| `itern` | Number of outer Newton iterations per sub-sequence (default `50`). |
| `cg_tol` | Relative residual tolerance for the inner CG solve (default `1e-6`). |
| `cg_maxiter` | Maximum CG iterations per Newton step (default `200`). |

## Returns

A dictionary with three flattened arrays, each covering the portion of `data` actually processed (truncated to a multiple of `n`):

- `data` — the original input signal.
- `denoised_data` — the reconstructed signal from re-integrating the recovered derivative.
- `diff_data` — the recovered derivative estimate.

## Notes

- Any trailing samples that don't fill a full sub-sequence of length `n` are dropped.
- Choosing `n` involves a trade-off: larger sub-sequences capture longer-range structure but cost more per Newton/CG solve; smaller sub-sequences are cheaper but lose context across boundaries.
