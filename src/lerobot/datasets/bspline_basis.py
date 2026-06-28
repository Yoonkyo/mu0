"""Cubic B-spline basis matrix for trajectory parameterisation.

Ported from ``temp/TraceAnything/trace_anything/trace_anything.py:49-99``.
We support ``n_ctrl in {4, 7, 10}`` because the precomputed knot vectors are
multiplicity-3-internal-knot Bezier-like cubics; other sizes would require
runtime knot generation.

The returned matrix ``N`` has shape ``(T, n_ctrl)`` and rows that sum to 1
(partition of unity). Use it as ``X = N @ P`` (forward render) or
``P = lstsq(N, X).solution`` (fit).
"""

from __future__ import annotations

import torch
from torch import Tensor

_DEGREE = 3

_PRECOMPUTED_KNOTS: dict[int, Tensor] = {
    4: torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    7: torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    ),
    10: torch.tensor(
        [
            0.0, 0.0, 0.0, 0.0,
            1 / 3, 1 / 3, 1 / 3,
            2 / 3, 2 / 3, 2 / 3,
            1.0, 1.0, 1.0, 1.0,
        ]
    ),
}


def _knot_diffs(n: int) -> tuple[Tensor, Tensor]:
    knots = _PRECOMPUTED_KNOTS[n]
    denom1 = torch.zeros(n, _DEGREE + 1)
    denom2 = torch.zeros(n, _DEGREE + 1)
    for k in range(_DEGREE + 1):
        for i in range(n):
            denom1[i, k] = knots[i + k] - knots[i] if i + k < len(knots) else 0.0
            denom2[i, k] = (
                knots[i + k + 1] - knots[i + 1]
                if i + k + 1 < len(knots)
                else 1.0
            )
    return denom1, denom2


_PRECOMPUTED_DENOMS: dict[int, tuple[Tensor, Tensor]] = {
    n: _knot_diffs(n) for n in (4, 7, 10)
}


def _basis_at(n_ctrl: int, t: Tensor) -> Tensor:
    knots = _PRECOMPUTED_KNOTS[n_ctrl].to(dtype=t.dtype, device=t.device)
    denom1, denom2 = (
        d.to(dtype=t.dtype, device=t.device)
        for d in _PRECOMPUTED_DENOMS[n_ctrl]
    )
    T = t.shape[0]
    basis = torch.zeros(T, n_ctrl, _DEGREE + 1, dtype=t.dtype, device=t.device)

    for i in range(n_ctrl):
        if i == n_ctrl - 1:
            basis[:, i, 0] = ((knots[i] <= t) & (t <= knots[i + 1])).to(t.dtype)
        else:
            basis[:, i, 0] = ((knots[i] <= t) & (t < knots[i + 1])).to(t.dtype)

    for k in range(1, _DEGREE + 1):
        for i in range(n_ctrl):
            term1 = torch.zeros_like(t)
            term2 = torch.zeros_like(t)
            if denom1[i, k] > 0:
                term1 = ((t - knots[i]) / denom1[i, k]) * basis[:, i, k - 1]
            if denom2[i, k] > 0 and i + 1 < n_ctrl:
                term2 = (
                    (knots[i + k + 1] - t) / denom2[i, k]
                ) * basis[:, i + 1, k - 1]
            basis[:, i, k] = term1 + term2

    return basis[:, :, _DEGREE]


def build_bspline_basis(
    n_ctrl: int,
    horizon: int,
    *,
    include_anchor: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Cubic B-spline basis matrix at uniform-in-t sample points.

    Args:
        n_ctrl: number of control points (must be in ``{4, 7, 10}``).
        horizon: number of future timesteps ``H``.
        include_anchor: when True (default), prepend a row at ``t = 0`` so the
            first row evaluates the spline at the anchor (zero in scaled-delta
            space). Returned shape is ``(H + 1, n_ctrl)`` and rows ``[1:]``
            correspond to ``traj_future[:, 0..H-1]`` at ``t_i = (i + 1) / H``.
            When False, returns ``(H, n_ctrl)`` with rows at ``t_i = (i + 1) / H``.

    Returns:
        ``Tensor`` of shape ``(T, n_ctrl)`` on ``device`` with given ``dtype``.
    """
    if n_ctrl not in _PRECOMPUTED_KNOTS:
        raise ValueError(
            f"n_ctrl must be in {{4, 7, 10}}; got {n_ctrl}. "
            "Other sizes would require runtime knot generation."
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")

    if include_anchor:
        t = torch.arange(horizon + 1, dtype=dtype, device=device) / horizon
    else:
        t = (torch.arange(horizon, dtype=dtype, device=device) + 1) / horizon
    return _basis_at(n_ctrl, t)
