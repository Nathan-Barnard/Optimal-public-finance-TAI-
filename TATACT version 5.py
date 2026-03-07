#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# SECTION 1 — Core Infrastructure (Plan 2.3 / Plan 4.2)
# Grid + Params/Primitives + Masks + Gatekeepers + ω Quarantine
# + Policy-array utilities
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple, Optional, Iterable
import numpy as np

NEG_INF = -1.0e300


# ============================================================
# GRID
# ============================================================

def make_interior_mask(nx: int, ny: int, buffer: int = 1) -> np.ndarray:
    """
    Interior buffer mask:
      - False on the outer `buffer` ring(s)
      - True in the interior

    Special case:
      - buffer == 0 returns an all-True mask (no buffer).
    """
    if buffer < 0:
        raise ValueError("buffer must be nonnegative.")
    if buffer == 0:
        return np.ones((nx, ny), dtype=bool)

    if nx <= 2 * buffer or ny <= 2 * buffer:
        raise ValueError("Grid too small for requested interior buffer.")

    m = np.ones((nx, ny), dtype=bool)
    m[:buffer, :] = False
    m[-buffer:, :] = False
    m[:, :buffer] = False
    m[:, -buffer:] = False
    return m


@dataclass
class Grid:
    """
    State grid for x = (k, L).
    k: shape (Nx,)
    L: shape (Ny,)
    interior_mask: shape (Nx,Ny), True on B^∘
    """
    k: np.ndarray
    L: np.ndarray
    interior_mask: Optional[np.ndarray] = None
    interior_buffer: int = 1

    def __post_init__(self):
        self.k = np.asarray(self.k, dtype=float).reshape(-1)
        self.L = np.asarray(self.L, dtype=float).reshape(-1)

        self.Nx = int(self.k.size)
        self.Ny = int(self.L.size)
        self.N = int(self.Nx * self.Ny)

        if self.Nx < 2 or self.Ny < 2:
            raise ValueError("Need at least 2 grid points in each dimension.")

        # Monotone + uniform spacing checks (code assumes uniform grids)
        dk_vec = np.diff(self.k)
        dL_vec = np.diff(self.L)

        if np.any(dk_vec <= 0.0) or np.any(dL_vec <= 0.0):
            raise ValueError("Grid must be strictly increasing in both dimensions.")

        self.dk = float(dk_vec[0])
        self.dL = float(dL_vec[0])

        # Uniformity validation (light, but catches accidental nonuniform grids)
        if self.Nx > 2 and not np.allclose(dk_vec, self.dk):
            raise ValueError("Non-uniform k grid detected; uniform dk assumption violated.")
        if self.Ny > 2 and not np.allclose(dL_vec, self.dL):
            raise ValueError("Non-uniform L grid detected; uniform dL assumption violated.")

        if self.interior_mask is None:
            self.interior_mask = make_interior_mask(self.Nx, self.Ny, buffer=self.interior_buffer)
        else:
            self.interior_mask = np.asarray(self.interior_mask, dtype=bool)
            if self.interior_mask.shape != (self.Nx, self.Ny):
                raise ValueError("interior_mask has wrong shape.")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.Nx, self.Ny)


def flatten(node: Tuple[int, int], grid: Grid) -> int:
    i, j = node
    return int(i * grid.Ny + j)


def unflatten(idx: int, grid: Grid) -> Tuple[int, int]:
    i = int(idx // grid.Ny)
    j = int(idx % grid.Ny)
    return (i, j)


def iter_nodes_where_legacy_a(mask: np.ndarray) -> Iterator[Tuple[int, int]]:
    ii, jj = np.where(mask)
    for i, j in zip(ii, jj):
        yield (int(i), int(j))


# ============================================================
# PARAMETERS / PRIMITIVES
# ============================================================

@dataclass(frozen=True)
class Par:
    rho: float


@dataclass
class Prim:
    """
    Primitive control bounds and candidate grids.
    """
    tau_grid: np.ndarray
    h_grid: np.ndarray
    T_min: float
    T_max: float
    tau_min: float = 0.0
    tau_max: float = 1.0
    h_min: Optional[float] = None
    h_max: Optional[float] = None

    def __post_init__(self):
        self.tau_grid = np.asarray(self.tau_grid, dtype=float).reshape(-1)
        self.h_grid   = np.asarray(self.h_grid, dtype=float).reshape(-1)
        if self.tau_grid.size == 0 or self.h_grid.size == 0:
            raise ValueError("tau_grid and h_grid must be non-empty.")

        self.T_min = float(self.T_min)
        self.T_max = float(self.T_max)
        if not (self.T_min <= self.T_max):
            raise ValueError("Require T_min <= T_max.")

        self.tau_min = float(self.tau_min)
        self.tau_max = float(self.tau_max)
        if not (self.tau_min <= self.tau_max):
            raise ValueError("Require tau_min <= tau_max.")

        if self.h_min is None:
            self.h_min = float(np.min(self.h_grid))
        if self.h_max is None:
            self.h_max = float(np.max(self.h_grid))
        self.h_min = float(self.h_min)
        self.h_max = float(self.h_max)
        if not (self.h_min <= self.h_max):
            raise ValueError("Require h_min <= h_max.")

        # NEW: ensure candidate grids are consistent with declared bounds
        tol = 1e-12
        if np.any(self.tau_grid < self.tau_min - tol) or np.any(self.tau_grid > self.tau_max + tol):
            raise ValueError("tau_grid contains values outside [tau_min, tau_max].")
        if np.any(self.h_grid < self.h_min - tol) or np.any(self.h_grid > self.h_max + tol):
            raise ValueError("h_grid contains values outside [h_min, h_max].")


# ============================================================
# MASK / NORM UTILITIES
# ============================================================

def max_norm_on_mask(arr: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(arr)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != arr.shape:
        raise ValueError("mask and arr must have same shape.")
    if not np.any(mask):
        return 0.0

    v = np.abs(arr[mask])
    if v.size == 0:
        return 0.0

    m = float(np.nanmax(v))
    if not np.isfinite(m):
        raise RuntimeError("max_norm_on_mask: non-finite (NaN/Inf) encountered on the masked region.")
    return m


def masks_equal(A: np.ndarray, B: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(A, dtype=bool), np.asarray(B, dtype=bool)))


def masks_stable(history: list[dict], window: int = 3, min_size: int = 10) -> bool:
    """
    Simple stability test: mask sizes constant over last `window` iterations
    and not trivially collapsed.
    """
    if len(history) < window:
        return False
    s1 = [int(h["size_M1"]) for h in history[-window:]]
    s0 = [int(h["size_M0"]) for h in history[-window:]]
    return (min(s1) >= min_size and min(s0) >= min_size and (max(s1) == min(s1)) and (max(s0) == min(s0)))


# ============================================================
# ONE-CELL INWARD GATEKEEPER (VECTOR + NODE) WITH HARD WALLS
# ============================================================

def inward_one_cell(active: np.ndarray,
                    kdot: np.ndarray,
                    Ldot: np.ndarray,
                    eps: float = 0.0) -> np.ndarray:
    """
    Vector gate: a node is admissible if whenever drift points to a neighbour,
    that neighbour is active. Hard wall: neighbours off-grid treated as inactive.

    NaN/Inf-safe: if drift is not finite, gate returns False at that node.
    """
    active = np.asarray(active, dtype=bool)
    kdot = np.asarray(kdot, dtype=float)
    Ldot = np.asarray(Ldot, dtype=float)

    if active.shape != kdot.shape or active.shape != Ldot.shape:
        raise ValueError("active, kdot, Ldot must have the same shape.")

    # Neighbour-active masks with hard-wall fill=False
    east  = np.zeros_like(active); east[:-1, :]  = active[1:, :]
    west  = np.zeros_like(active); west[1:,  :]  = active[:-1, :]
    north = np.zeros_like(active); north[:, :-1] = active[:, 1:]
    south = np.zeros_like(active); south[:, 1:]  = active[:, :-1]

    need_e = (kdot >  eps)
    need_w = (kdot < -eps)
    need_n = (Ldot >  eps)
    need_s = (Ldot < -eps)

    ok = (~need_e | east) & (~need_w | west) & (~need_n | north) & (~need_s | south)
    ok &= active

    # Reject non-finite drift (prevents NaN comparisons from silently passing)
    finite = np.isfinite(kdot) & np.isfinite(Ldot)
    ok &= finite
    return ok


def inward_one_cell_node(active: np.ndarray,
                         node: Tuple[int, int],
                         kdot: float,
                         Ldot: float,
                         eps: float = 0.0) -> bool:
    """
    Scalar gate with hard grid walls and active-set walls.
    NaN/Inf drift -> False.
    """
    if not (np.isfinite(kdot) and np.isfinite(Ldot)):
        return False

    active = np.asarray(active, dtype=bool)
    i, j = node
    Nx, Ny = active.shape

    # Hard grid walls (no wraparound / no off-grid step)
    if kdot >  eps and i == Nx - 1: return False
    if kdot < -eps and i == 0:      return False
    if Ldot >  eps and j == Ny - 1: return False
    if Ldot < -eps and j == 0:      return False

    # Active-set walls
    if kdot >  eps and not active[i + 1, j]: return False
    if kdot < -eps and not active[i - 1, j]: return False
    if Ldot >  eps and not active[i, j + 1]: return False
    if Ldot < -eps and not active[i, j - 1]: return False

    return bool(active[i, j])


# ============================================================
# ω QUARANTINE: NEAREST-NEIGHBOUR FILL (PURE NUMPY)
# ============================================================

def quarantine_fill_nearest(arr: np.ndarray,
                            valid_mask: np.ndarray,
                            *,
                            max_iter: Optional[int] = None) -> np.ndarray:
    """
    Fill arr[~valid_mask] with nearest (Manhattan) valid neighbour values.
    Deterministic priority: up, down, left, right.

    Uses np.roll internally but is safe because take_* masks never include boundaries.
    """
    arr = np.asarray(arr, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if arr.shape != valid_mask.shape:
        raise ValueError("arr and valid_mask must have same shape.")
    if not np.any(valid_mask):
        raise ValueError("valid_mask has no True entries; cannot fill.")

    out = arr.copy()
    filled = valid_mask.copy()

    if max_iter is None:
        # Upper bound on manhattan diameter (loose)
        max_iter = int(arr.shape[0] + arr.shape[1] + 5)
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative.")

    for _ in range(max_iter):
        if filled.all():
            break

        # neighbour availability (hard walls)
        up    = np.zeros_like(filled); up[:-1, :]   = filled[1:, :]
        down  = np.zeros_like(filled); down[1:, :]  = filled[:-1, :]
        left  = np.zeros_like(filled); left[:, 1:]  = filled[:, :-1]
        right = np.zeros_like(filled); right[:, :-1]= filled[:, 1:]

        can_fill = (~filled) & (up | down | left | right)
        if not np.any(can_fill):
            break

        # Priority order: up, down, left, right
        take_up = can_fill & up
        if np.any(take_up):
            out[take_up] = np.roll(out, -1, axis=0)[take_up]
            filled[take_up] = True

        take_down = can_fill & (~take_up) & down
        if np.any(take_down):
            out[take_down] = np.roll(out, 1, axis=0)[take_down]
            filled[take_down] = True

        take_left = can_fill & (~take_up) & (~take_down) & left
        if np.any(take_left):
            out[take_left] = np.roll(out, 1, axis=1)[take_left]
            filled[take_left] = True

        take_right = can_fill & (~take_up) & (~take_down) & (~take_left) & right
        if np.any(take_right):
            out[take_right] = np.roll(out, -1, axis=1)[take_right]
            filled[take_right] = True

    if not filled.all():
        raise RuntimeError("quarantine_fill_nearest did not fill entire array (disconnected mask).")

    return out


# ============================================================
# POLICY ARRAY UTILITIES (SELF-CONTAINED)
# ============================================================

Policy = Dict[str, np.ndarray]  # keys: "tau", "h", "T"

def empty_policy_like_legacy_a(mask: np.ndarray) -> Policy:
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    return {
        "tau": np.full(shape, np.nan, dtype=float),
        "h":   np.full(shape, np.nan, dtype=float),
        "T":   np.full(shape, np.nan, dtype=float),
    }

def mask_policy_legacy_a(u: Policy, mask: np.ndarray) -> Policy:
    mask = np.asarray(mask, dtype=bool)
    out = {k: np.asarray(v, dtype=float).copy() for k, v in u.items()}
    for key in ("tau", "h", "T"):
        if key not in out:
            raise KeyError(f"Policy missing key '{key}'")
        if out[key].shape != mask.shape:
            raise ValueError("Policy arrays and mask must have same shape.")
        out[key][~mask] = np.nan
    return out

def blend_and_project_on_mask(u_old: Policy,
                              u_targ: Policy,
                              eta: float,
                              mask: np.ndarray,
                              prim: Prim) -> Policy:
    """
    Plan 2.3 NaN-safe damping:
      blend only on `mask`, keep u_old elsewhere.
    """
    mask = np.asarray(mask, dtype=bool)
    out: Policy = {}
    for key in ("tau", "h", "T"):
        out[key] = np.where(mask,
                            (1.0 - eta) * u_old[key] + eta * u_targ[key],
                            u_old[key]).astype(float)

    out["tau"] = np.clip(out["tau"], prim.tau_min, prim.tau_max)
    out["h"]   = np.clip(out["h"],   prim.h_min,   prim.h_max)
    out["T"]   = np.clip(out["T"],   prim.T_min,   prim.T_max)
    return out

def select_blend_or_snap(u_blend: Policy, u_targ: Policy, ok_mask: np.ndarray) -> Policy:
    ok_mask = np.asarray(ok_mask, dtype=bool)
    out: Policy = {}
    for key in ("tau", "h", "T"):
        out[key] = np.where(ok_mask, u_blend[key], u_targ[key]).astype(float)
    return out

def policy_supnorm(u_new: Policy, u_old: Policy, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0

    d = 0.0
    for key in ("tau", "h", "T"):
        diff = np.abs(u_new[key] - u_old[key])
        vals = diff[mask]
        if vals.size == 0:
            continue
        m = float(np.nanmax(vals))
        if not np.isfinite(m):
            raise RuntimeError(f"policy_supnorm: non-finite (NaN/Inf) diff encountered on masked region for key '{key}'.")
        d = max(d, m)
    return d

def coarse_rescue_candidate_set(prim: Prim) -> Iterable[Tuple[float, float, float]]:
    """
    Small candidate set used in viability peeling (fast existence checks).
    """
    tau_mid = float(np.median(prim.tau_grid))
    h_mid   = float(np.median(prim.h_grid))
    taus = [float(prim.tau_min), tau_mid, float(prim.tau_max)]
    hs   = [float(prim.h_min),   h_mid,   float(prim.h_max)]
    Ts   = [float(prim.T_min), 0.0, float(prim.T_max)]
    for tau in taus:
        for h in hs:
            for T in Ts:
                yield (tau, h, T)


# ============================================================
# QUICK SANITY CHECKS (OPTIONAL TO RUN)
# ============================================================

def _sanity_check_section_1():
    # buffer=0 should be all-True
    m0 = make_interior_mask(5, 6, buffer=0)
    assert m0.shape == (5, 6) and bool(m0.all())

    # tiny grid
    k = np.linspace(1.0, 2.0, 5)
    L = np.linspace(-0.5, 0.5, 6)
    g = Grid(k, L)  # auto interior mask
    assert g.interior_mask.shape == (g.Nx, g.Ny)

    # inward gate should fail if we try to step off-grid
    active = np.ones((g.Nx, g.Ny), dtype=bool)
    assert inward_one_cell_node(active, (g.Nx-1, 3), kdot=1.0, Ldot=0.0, eps=0.0) is False
    assert inward_one_cell_node(active, (0, 3),      kdot=-1.0, Ldot=0.0, eps=0.0) is False

    # quarantine fill
    arr = np.zeros((g.Nx, g.Ny))
    arr[2, 2] = 7.0
    valid = np.zeros_like(arr, dtype=bool)
    valid[2, 2] = True
    filled = quarantine_fill_nearest(arr, valid)
    assert np.allclose(filled, 7.0)

# Uncomment to run:
# _sanity_check_section_1()  # moved under __main__ guard


# In[2]:


# ============================================================
# SECTION 2 — Masked Finite Differences (Plan 2.3)
# Computes forward/backward derivatives on an active set, with
# NaN marking where the stencil would cross inactive nodes.
# ============================================================

from typing import Tuple, Optional
import numpy as np


def masked_upwind_derivatives(
    J: np.ndarray,
    active: np.ndarray,
    grid: Grid,
    *,
    invalid: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute masked forward/backward finite differences of J on the 2D grid.

    Returns:
      Jk_f, Jk_b, JL_f, JL_b  (all shape (Nx,Ny))

    Convention:
      Jk_f[i,j] = (J[i+1,j] - J[i,j]) / dk   if active[i,j] & active[i+1,j]
      Jk_b[i,j] = (J[i,j] - J[i-1,j]) / dk   if active[i,j] & active[i-1,j]
      JL_f[i,j] = (J[i,j+1] - J[i,j]) / dL   if active[i,j] & active[i,j+1]
      JL_b[i,j] = (J[i,j] - J[i,j-1]) / dL   if active[i,j] & active[i,j-1]

    If the needed neighbour is inactive OR J is non-finite at either node,
    derivative is set to `invalid` (default: np.nan).
    """
    J = np.asarray(J, dtype=float)
    active = np.asarray(active, dtype=bool)

    if J.shape != active.shape:
        raise ValueError("J and active must have the same shape.")
    if J.shape != grid.shape:
        raise ValueError("J shape must match grid.shape.")

    dk, dL = float(grid.dk), float(grid.dL)

    Jk_f = np.full(J.shape, invalid, dtype=float)
    Jk_b = np.full(J.shape, invalid, dtype=float)
    JL_f = np.full(J.shape, invalid, dtype=float)
    JL_b = np.full(J.shape, invalid, dtype=float)

    # --- k-direction diffs ---
    diff_k = (J[1:, :] - J[:-1, :]) / dk  # shape (Nx-1, Ny)

    m_k = active[:-1, :] & active[1:, :] & np.isfinite(J[:-1, :]) & np.isfinite(J[1:, :])

    # forward derivative at i uses (i -> i+1): store on rows 0..Nx-2
    Jk_f[:-1, :][m_k] = diff_k[m_k]

    # backward derivative at i uses (i-1 -> i): store on rows 1..Nx-1
    Jk_b[1:, :][m_k] = diff_k[m_k]

    # --- L-direction diffs ---
    diff_L = (J[:, 1:] - J[:, :-1]) / dL  # shape (Nx, Ny-1)

    m_L = active[:, :-1] & active[:, 1:] & np.isfinite(J[:, :-1]) & np.isfinite(J[:, 1:])

    # forward derivative at j uses (j -> j+1): store on cols 0..Ny-2
    JL_f[:, :-1][m_L] = diff_L[m_L]

    # backward derivative at j uses (j-1 -> j): store on cols 1..Ny-1
    JL_b[:, 1:][m_L] = diff_L[m_L]

    return Jk_f, Jk_b, JL_f, JL_b


def upwind_grad_components(
    Jk_f: np.ndarray, Jk_b: np.ndarray,
    JL_f: np.ndarray, JL_b: np.ndarray,
    kdot: np.ndarray, Ldot: np.ndarray,
    *,
    eps_drift: float = 0.0,
    active: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the upwind gradient components consistent with drift:

      if kdot >  eps: use Jk_f
      if kdot < -eps: use Jk_b
      else:           0

    Same for Ldot with JL_f/JL_b.

    Returns:
      Jk_up, JL_up, bad

    bad flags nodes where drift requires a derivative but the chosen derivative
    is non-finite (NaN/Inf), or where drift itself is non-finite.

    If `active` is provided, bad is restricted to active nodes.
    """
    Jk_f = np.asarray(Jk_f, dtype=float)
    Jk_b = np.asarray(Jk_b, dtype=float)
    JL_f = np.asarray(JL_f, dtype=float)
    JL_b = np.asarray(JL_b, dtype=float)
    kdot = np.asarray(kdot, dtype=float)
    Ldot = np.asarray(Ldot, dtype=float)

    if not (Jk_f.shape == Jk_b.shape == JL_f.shape == JL_b.shape == kdot.shape == Ldot.shape):
        raise ValueError("All derivative arrays and drift arrays must have the same shape.")

    Jk_up = np.zeros_like(kdot, dtype=float)
    JL_up = np.zeros_like(Ldot, dtype=float)

    pos_k = kdot > eps_drift
    neg_k = kdot < -eps_drift
    pos_L = Ldot > eps_drift
    neg_L = Ldot < -eps_drift

    Jk_up[pos_k] = Jk_f[pos_k]
    Jk_up[neg_k] = Jk_b[neg_k]
    JL_up[pos_L] = JL_f[pos_L]
    JL_up[neg_L] = JL_b[neg_L]

    bad = np.zeros_like(pos_k, dtype=bool)
    bad |= (pos_k | neg_k) & (~np.isfinite(Jk_up))
    bad |= (pos_L | neg_L) & (~np.isfinite(JL_up))
    bad |= ~np.isfinite(kdot) | ~np.isfinite(Ldot)

    if active is not None:
        active = np.asarray(active, dtype=bool)
        if active.shape != bad.shape:
            raise ValueError("active mask must have the same shape as drift arrays.")
        bad &= active

    return Jk_up, JL_up, bad


def _sanity_check_section_2():
    k = np.linspace(1.0, 2.0, 5)
    L = np.linspace(-0.5, 0.5, 6)
    g = Grid(k, L)

    # Define a simple linear function J = 2k + 3L
    K, LL = np.meshgrid(g.k, g.L, indexing="ij")
    J = 2.0 * K + 3.0 * LL

    active = g.interior_mask.copy()

    Jk_f, Jk_b, JL_f, JL_b = masked_upwind_derivatives(J, active, g)

    # Test only where derivatives are actually defined (finite)
    m1 = active & np.isfinite(Jk_f)
    m2 = active & np.isfinite(Jk_b)
    m3 = active & np.isfinite(JL_f)
    m4 = active & np.isfinite(JL_b)

    k_ok = (np.max(np.abs(Jk_f[m1] - 2.0)) < 1e-12 if np.any(m1) else True) and \
           (np.max(np.abs(Jk_b[m2] - 2.0)) < 1e-12 if np.any(m2) else True)

    L_ok = (np.max(np.abs(JL_f[m3] - 3.0)) < 1e-12 if np.any(m3) else True) and \
           (np.max(np.abs(JL_b[m4] - 3.0)) < 1e-12 if np.any(m4) else True)

    print("Section 2 check:",
          "Jk derivatives OK" if k_ok else "Jk derivatives FAIL",
          "|",
          "JL derivatives OK" if L_ok else "JL derivatives FAIL")

# Uncomment to run:
# _sanity_check_section_2()  # moved under __main__ guard


# In[3]:


# ============================================================
# SECTION 3 — Masked Generator Assembly + Linear HJB Solves (Plan 2.3 / Plan 4.2)
#
# Core idea:
#   - Build the upwind transport generator A ONLY on the active subspace M
#     (so there are no columns pointing to ~M => no ghost leakage).
#   - Use the SAME eps_drift threshold as the inward gatekeeper.
#   - Solve linear systems on the active submatrix and embed back to full grid.
#
# Dependencies: Section 1 utilities (Grid, flatten, iter_nodes_where, inward_one_cell_node)
# Requires: scipy.sparse
# ============================================================

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# ACTIVE SUBSPACE INDEXING
# ============================================================

@dataclass(frozen=True)
class ActiveIndex:
    """
    Index mapping for an active mask M (bool (Nx,Ny)).

    idx_full: flat indices into the full grid vector (length n_active)
    inv_full: inverse map of length grid.N (full flat index -> [0..n_active-1] or -1)
    n_active: number of active nodes
    """
    idx_full: np.ndarray
    inv_full: np.ndarray
    n_active: int


def build_active_index(active: np.ndarray, grid: Grid) -> ActiveIndex:
    active = np.asarray(active, dtype=bool)
    if active.shape != grid.shape:
        raise ValueError("active mask must have shape grid.shape")

    idx_full = np.flatnonzero(active.ravel(order="C")).astype(np.int64)
    inv_full = -np.ones(grid.N, dtype=np.int64)
    inv_full[idx_full] = np.arange(idx_full.size, dtype=np.int64)

    return ActiveIndex(idx_full=idx_full, inv_full=inv_full, n_active=int(idx_full.size))


def embed_active_to_full(
    x_act: np.ndarray,
    act: ActiveIndex,
    grid: Grid,
    *,
    anchor: float = np.nan,
) -> np.ndarray:
    """
    Embed an active-space vector x_act (length n_active) into a full-grid array (Nx,Ny).
    """
    x_act = np.asarray(x_act, dtype=float).reshape(-1)
    if x_act.size != act.n_active:
        raise ValueError("x_act has wrong length for this ActiveIndex")

    full = np.full(grid.N, float(anchor), dtype=float)
    full[act.idx_full] = x_act
    return full.reshape(grid.shape, order="C")


def restrict_full_to_active(
    x_full: np.ndarray,
    act: ActiveIndex,
    grid: Grid,
    *,
    check_finite: bool = True,
) -> np.ndarray:
    """
    Restrict a full-grid array (Nx,Ny) to the active-space vector (length n_active).
    """
    x_full = np.asarray(x_full, dtype=float)
    if x_full.shape != grid.shape:
        raise ValueError("x_full must have shape grid.shape")

    vec = x_full.ravel(order="C")[act.idx_full]
    if check_finite and (not np.isfinite(vec).all()):
        raise RuntimeError("restrict_full_to_active found non-finite values on active set")
    return vec


# ============================================================
# NODE EVALUATOR INTERFACE
# ============================================================

@dataclass(frozen=True)
class NodeDriftFlow:
    """
    What the masked generator needs from your model at each node:
      - deterministic drift components (kdot, Ldot)
      - instantaneous flow payoff (flow) entering rho*J = flow + A J
    """
    kdot: float
    Ldot: float
    flow: float


NodeEvaluator = Callable[[Tuple[int, int]], NodeDriftFlow]


# ============================================================
# MASKED GENERATOR ASSEMBLY (EPS-CONSISTENT, NO GHOST LEAKAGE)
# ============================================================

def build_masked_system_2_3(
    grid: Grid,
    active: np.ndarray,
    node_eval: NodeEvaluator,
    *,
    eps_drift: float = 0.0,
    check_inward: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, ActiveIndex]:
    """
    Assemble (A, f) on the active subspace for:
        rho * J = f + A J
    where A is a generator induced by upwind drift transitions.

    - A is (n_active x n_active) sparse CSR.
    - f is (n_active,) dense.
    - No columns outside active exist (NO ghost leakage possible).
    - Transition direction tests use the SAME eps_drift as inward gating.

    If check_inward=True, we assert that the drift respects the one-cell gate
    on 'active' at every active node (should be guaranteed by improvement+closure).
    """
    if eps_drift < 0.0:
        raise ValueError("eps_drift must be nonnegative.")

    active = np.asarray(active, dtype=bool)
    if active.shape != grid.shape:
        raise ValueError("active mask must have shape grid.shape")

    act = build_active_index(active, grid)
    if act.n_active == 0:
        raise ValueError("Active set is empty; cannot build system")

    Nx, Ny = grid.shape
    dk = float(grid.dk)
    dL = float(grid.dL)
    if not (dk > 0.0 and dL > 0.0):
        raise ValueError("Grid spacings dk and dL must be positive")

    # COO triplets
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    f = np.zeros(act.n_active, dtype=float)

    for node in iter_nodes_where(active):
        i, j = node

        p_full = flatten(node, grid)
        p = int(act.inv_full[p_full])
        if p < 0:
            raise RuntimeError("Internal error: active node not in active index")

        nd = node_eval(node)
        kdot = float(nd.kdot)
        Ldot = float(nd.Ldot)
        flow = float(nd.flow)

        if not (np.isfinite(kdot) and np.isfinite(Ldot) and np.isfinite(flow)):
            raise RuntimeError(f"Non-finite node_eval output at node={node}: {nd}")

        # Optional consistency assertion (recommended while debugging)
        if check_inward:
            if not inward_one_cell_node(active, node, kdot, Ldot, eps=eps_drift):
                raise RuntimeError(
                    f"Active policy violates inward gate at node={node} "
                    f"(kdot={kdot}, Ldot={Ldot}, eps={eps_drift})."
                )

        out = 0.0  # row outflow sum (positive)

        # ----- k direction (epsilon-consistent) -----
        if kdot > eps_drift:
            # Hard-wall safety (prevents IndexError if active touches the outer boundary)
            if i == Nx - 1:
                raise RuntimeError("Off-grid east step at boundary node; mask/gate inconsistent.")
            q_full = p_full + Ny  # (i+1, j)
            q = int(act.inv_full[q_full])
            if q < 0:
                raise RuntimeError("Transition to inactive neighbour (east). Gatekeeper inconsistency.")
            a = kdot / dk
            rows.append(p); cols.append(q); data.append(a)
            out += a

        elif kdot < -eps_drift:
            if i == 0:
                raise RuntimeError("Off-grid west step at boundary node; mask/gate inconsistent.")
            q_full = p_full - Ny  # (i-1, j)
            q = int(act.inv_full[q_full])
            if q < 0:
                raise RuntimeError("Transition to inactive neighbour (west). Gatekeeper inconsistency.")
            a = (-kdot) / dk
            rows.append(p); cols.append(q); data.append(a)
            out += a

        # ----- L direction (epsilon-consistent) -----
        if Ldot > eps_drift:
            if j == Ny - 1:
                raise RuntimeError("Off-grid north step at boundary node; mask/gate inconsistent.")
            q_full = p_full + 1  # (i, j+1)
            q = int(act.inv_full[q_full])
            if q < 0:
                raise RuntimeError("Transition to inactive neighbour (north). Gatekeeper inconsistency.")
            a = Ldot / dL
            rows.append(p); cols.append(q); data.append(a)
            out += a

        elif Ldot < -eps_drift:
            if j == 0:
                raise RuntimeError("Off-grid south step at boundary node; mask/gate inconsistent.")
            q_full = p_full - 1  # (i, j-1)
            q = int(act.inv_full[q_full])
            if q < 0:
                raise RuntimeError("Transition to inactive neighbour (south). Gatekeeper inconsistency.")
            a = (-Ldot) / dL
            rows.append(p); cols.append(q); data.append(a)
            out += a

        # diagonal so rows sum to 0
        if out != 0.0:
            rows.append(p); cols.append(p); data.append(-out)

        f[p] = flow

    A = sp.coo_matrix(
        (np.asarray(data, dtype=float),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(act.n_active, act.n_active),
    ).tocsr()

    return A, f, act


# ============================================================
# GENERATOR SANITY CHECKS (OPTIONAL BUT VERY USEFUL)
# ============================================================

def check_generator_properties(A: sp.spmatrix, *, tol: float = 1e-12) -> None:
    """
    Basic generator checks for upwind transport matrices:
      - off-diagonals >= 0
      - diagonals <= 0
      - row sums approx 0
    """
    A = A.tocsr()
    diag = A.diagonal()
    if np.any(diag > tol):
        raise RuntimeError("Generator has positive diagonal entries (should be <= 0).")

    # Check off-diagonals nonnegative
    A_off = A.copy()
    A_off.setdiag(0.0)
    A_off.eliminate_zeros()
    if A_off.nnz > 0 and np.min(A_off.data) < -tol:
        raise RuntimeError("Generator has negative off-diagonal entries (should be >= 0).")

    # Row sum ~ 0 (allow slightly looser tolerance than elementwise checks)
    rs = np.asarray(A.sum(axis=1)).reshape(-1)
    row_tol = max(1e-9, 100.0 * tol)
    if np.max(np.abs(rs)) > row_tol:
        raise RuntimeError(f"Generator row sums not ~0; max|row_sum|={np.max(np.abs(rs))}")


# ============================================================
# LINEAR SOLVES ON ACTIVE SUBSPACE
# ============================================================

def solve_hjb_on_active(
    A: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    rho: float,
    lam: float = 0.0,
    solver: str = "spsolve",
) -> np.ndarray:
    """
    Solve on active subspace:
      ((rho + lam) I - A) J = rhs

    Returns:
      J_act (length n_active)
    """
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if lam < 0.0:
        raise ValueError("lam must be nonnegative")

    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    n = int(A.shape[0])
    if A.shape != (n, n) or rhs.size != n:
        raise ValueError("Dimension mismatch in solve_hjb_on_active")

    if not np.isfinite(rhs).all():
        raise RuntimeError("Non-finite rhs in solve_hjb_on_active")

    I = sp.eye(n, format="csr", dtype=float)
    LHS = (rho + lam) * I - A  # M-matrix structure for upwind transport

    if solver == "spsolve":
        J_act = spla.spsolve(LHS.tocsc(), rhs)
        J_act = np.asarray(J_act, dtype=float)
    elif solver == "bicgstab":
        J_act, info = spla.bicgstab(LHS, rhs, atol=0.0, rtol=1e-12, maxiter=10_000)
        if info != 0:
            raise RuntimeError(f"bicgstab failed with info={info}")
        J_act = np.asarray(J_act, dtype=float)
    else:
        raise ValueError("Unknown solver; use 'spsolve' or 'bicgstab'")

    if not np.isfinite(J_act).all():
        raise RuntimeError("Non-finite solution returned by linear solver")

    return J_act


def solve_and_embed_hjb(
    grid: Grid,
    A: sp.csr_matrix,
    rhs: np.ndarray,
    act: ActiveIndex,
    *,
    rho: float,
    lam: float = 0.0,
    anchor: float = np.nan,
    solver: str = "spsolve",
) -> np.ndarray:
    """
    Convenience: solve ((rho+lam)I - A) J = rhs on active subspace, then embed into (Nx,Ny).
    """
    J_act = solve_hjb_on_active(A, rhs, rho=rho, lam=lam, solver=solver)
    return embed_active_to_full(J_act, act, grid, anchor=anchor)


# ============================================================
# OPTIONAL SANITY CHECK (THIS ONE PRINTS)
# ============================================================

def _sanity_check_section_3():
    # Tiny grid and a drift field that is inward on interior_mask
    k = np.linspace(1.0, 2.0, 6)
    L = np.linspace(-0.5, 0.5, 7)
    g = Grid(k, L)

    active = g.interior_mask.copy()
    Nx, Ny = g.shape

    # Drift points toward the center of the active region (so it is inward everywhere on 'active')
    def node_eval(node):
        i, j = node
        kdot = 0.1 if i < Nx // 2 else -0.1
        Ldot = 0.2 if j < Ny // 2 else -0.2
        return NodeDriftFlow(kdot=kdot, Ldot=Ldot, flow=1.0)

    A, f, act = build_masked_system_2_3(g, active, node_eval, eps_drift=0.0, check_inward=True)
    check_generator_properties(A)

    J = solve_and_embed_hjb(g, A, f, act, rho=0.05, lam=0.0, anchor=np.nan)

    print("Section 3 check:",
          "A shape =", A.shape,
          "| active nodes =", act.n_active,
          "| J finite on active =", np.isfinite(J[active]).all(),
          "| row-sum max abs =", float(np.max(np.abs(np.asarray(A.sum(axis=1)).reshape(-1)))))

# Uncomment to run:
# _sanity_check_section_3()


# In[4]:


# ============================================================
# RESTORED SECTION — Viability Peel (Warm Peel + Fixpoint)
# ============================================================

from typing import Callable, Iterable, Optional, Tuple, List
import numpy as np

# Type aliases for model-specific callbacks:
StaticFeasibleNode = Callable[
    [Grid, Par, Prim, int, Tuple[int, int], float, float, float, np.ndarray], bool
]
DriftNode = Callable[
    [Grid, Par, Prim, int, Tuple[int, int], float, float, float, np.ndarray], Tuple[float, float]
]

def _materialize_peel_candidates(
    prim: Prim,
    candidates: Optional[Iterable[Tuple[float, float, float]]],
) -> List[Tuple[float, float, float]]:
    if candidates is None:
        cand_list = list(coarse_rescue_candidate_set(prim))
    elif isinstance(candidates, list):
        cand_list = candidates
    else:
        cand_list = list(candidates)

    if len(cand_list) == 0:
        raise ValueError("Candidate set is empty; peel cannot proceed.")
    return cand_list

def viability_peel_step(
    V: np.ndarray,
    base_active: np.ndarray,
    grid: Grid,
    par: Par,
    prim: Prim,
    omega: np.ndarray,
    s: int,
    *,
    static_feasible_node: StaticFeasibleNode,
    drift_node: DriftNode,
    candidates: Optional[Iterable[Tuple[float, float, float]]] = None,
    eps_drift: float = 1e-12,
) -> np.ndarray:
    
    V = np.asarray(V, dtype=bool)
    base_active = np.asarray(base_active, dtype=bool)
    omega = np.asarray(omega, dtype=float)

    active = V & base_active
    if not np.any(active):
        return np.zeros_like(V, dtype=bool)

    cand_list = _materialize_peel_candidates(prim, candidates)
    keep = np.zeros_like(V, dtype=bool)

    for node in iter_nodes_where(active):
        feasible = False

        for (tau, h, T) in cand_list:
            if not static_feasible_node(grid, par, prim, s, node, tau, h, T, omega):
                continue

            kdot, Ldot = drift_node(grid, par, prim, s, node, tau, h, T, omega)

            if inward_one_cell_node(active, node, kdot, Ldot, eps=eps_drift):
                feasible = True
                break

        keep[node] = feasible

    return keep & base_active

def viability_peel_warm(
    V: np.ndarray,
    base_active: np.ndarray,
    grid: Grid,
    par: Par,
    prim: Prim,
    omega: np.ndarray,
    s: int,
    *,
    static_feasible_node: StaticFeasibleNode,
    drift_node: DriftNode,
    candidates: Optional[Iterable[Tuple[float, float, float]]] = None,
    N_peel: int = 5,
    eps_drift: float = 1e-12,
    verbose: bool = False,
) -> np.ndarray:
    
    V_new = np.asarray(V, dtype=bool).copy()
    cand_list = _materialize_peel_candidates(prim, candidates)

    for it in range(N_peel):
        V_next = viability_peel_step(
            V_new, base_active, grid, par, prim, omega, s,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            candidates=cand_list,         
            eps_drift=eps_drift,
        )
        if masks_equal(V_next, V_new):
            break
        V_new = V_next

    return V_new

def viability_peel_to_fixpoint(
    V: np.ndarray,
    base_active: np.ndarray,
    grid: Grid,
    par: Par,
    prim: Prim,
    omega: np.ndarray,
    s: int,
    *,
    static_feasible_node: StaticFeasibleNode,
    drift_node: DriftNode,
    candidates: Optional[Iterable[Tuple[float, float, float]]] = None,
    eps_drift: float = 1e-12,
    max_iter: int = 500,
    verbose: bool = False,
) -> np.ndarray:
    
    V_new = np.asarray(V, dtype=bool).copy()
    cand_list = _materialize_peel_candidates(prim, candidates)

    for it in range(max_iter):
        V_next = viability_peel_step(
            V_new, base_active, grid, par, prim, omega, s,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            candidates=cand_list,         
            eps_drift=eps_drift,
        )
        if masks_equal(V_next, V_new):
            break
        V_new = V_next

    return V_new


# In[5]:


# ============================================================
# SECTION 5 — Policy Improvement + Prune Closure (PATCHED)  (Plan 2.3 / Plan 4.2)
#
# Patches vs your previous Section 5:
#   (1) Rescue transfer now has access to the ACTIVE mask and to which neighbour
#       directions are blocked at the node (so it can enforce the correct inward
#       inequality).
#   (2) If transfer_rule (Achdou–Moll style) is used, we ALSO try a small backup
#       T-set (e.g. {T_min, T_mid, T_max} or user-provided T_grid). This prevents
#       "false pruning" when the analytic proposal is infeasible/noisy at boundaries.
#   (3) Optional behaviour when no feasible control exists on a truncation boundary:
#       "raise" (debug / grid-too-small diagnostic) or "prune" (default).
#   (4) Hard fail if prune-closure is called with an empty mask (prevents silent
#       fake-convergence chains).
#
# You provide model-specific callbacks:
#   static_feasible_node(grid, par, prim, s, node, tau, h, T, omega) -> bool
#   node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega) -> (flow, kdot, Ldot)
#
# Optional:
#   transfer_rule(...) -> T_am
#   rescue_transfer(...) -> T_rescue or None
# ============================================================

from __future__ import annotations
from typing import Callable, Optional, Tuple, Dict, Iterable, Literal
import numpy as np

Policy = Dict[str, np.ndarray]

StaticFeasibleNode = Callable[[Grid, Par, Prim, int, tuple[int, int], float, float, float, np.ndarray], bool]
NodeFlowAndDrift   = Callable[[Grid, Par, Prim, int, tuple[int, int], float, float, float, np.ndarray],
                              Tuple[float, float, float]]

# Analytic transfer rule (e.g., Achdou–Moll): returns a scalar T
TransferRule = Callable[
    [Grid, Par, Prim, int, tuple[int, int], float, float,
     np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    float
]

# Blocked directions flags (east, west, north, south)
BlockedDirs = Tuple[bool, bool, bool, bool]

# Rescue transfer (binding inward correction): returns scalar T or None
# NOTE: now receives active mask and blocked directions
RescueTransfer = Callable[
    [Grid, Par, Prim, int, tuple[int, int], float, float, np.ndarray, np.ndarray, float, BlockedDirs],
    Optional[float]
]


def _upwind_scalar_deriv(drift: float, J_f: float, J_b: float, eps: float) -> float:
    """
    Select upwind derivative (scalar):
      if drift > eps   -> forward
      if drift < -eps  -> backward
      else             -> 0

    Returns np.nan if required derivative is non-finite.
    """
    if drift > eps:
        return J_f if np.isfinite(J_f) else np.nan
    if drift < -eps:
        return J_b if np.isfinite(J_b) else np.nan
    return 0.0


def _is_grid_boundary(node: tuple[int, int], grid: Grid) -> bool:
    i, j = node
    return (i == 0) or (i == grid.Nx - 1) or (j == 0) or (j == grid.Ny - 1)


def _blocked_directions(
    active: np.ndarray,
    node: tuple[int, int],
    kdot: float,
    Ldot: float,
    eps: float,
) -> BlockedDirs:
    """
    Identify which neighbour directions are "blocked" given drift and active set.

    Returns flags (block_e, block_w, block_n, block_s).
    True means: drift wants to move that way AND the neighbour is off-grid or inactive.
    """
    i, j = node
    Nx, Ny = active.shape

    block_e = (kdot > eps) and (i == Nx - 1 or (not active[i + 1, j]))
    block_w = (kdot < -eps) and (i == 0      or (not active[i - 1, j]))
    block_n = (Ldot > eps) and (j == Ny - 1 or (not active[i, j + 1]))
    block_s = (Ldot < -eps) and (j == 0      or (not active[i, j - 1]))
    return (block_e, block_w, block_n, block_s)


def _default_T_backup(prim: Prim, T_grid: Optional[np.ndarray]) -> np.ndarray:
    """
    Backup T candidates. Always within [T_min, T_max].
    If T_grid is provided, use it (clipped & unique). Else use {T_min, T_mid, T_max}.
    """
    if T_grid is None:
        T_mid = 0.5 * (prim.T_min + prim.T_max)
        T_bak = np.array([prim.T_min, T_mid, prim.T_max], dtype=float)
    else:
        T_bak = np.asarray(T_grid, dtype=float).reshape(-1)
        if T_bak.size == 0:
            raise ValueError("T_grid is empty.")
        T_bak = np.unique(np.clip(T_bak, prim.T_min, prim.T_max))
    return T_bak


def policy_improvement_gatekeep_legacy_a(
    grid: Grid,
    par: Par,
    prim: Prim,
    s: int,
    J: np.ndarray,
    omega: np.ndarray,
    active: np.ndarray,
    *,
    static_feasible_node: StaticFeasibleNode,
    node_flow_and_drift: NodeFlowAndDrift,
    eps_drift: float = 1e-12,
    # Transfer handling:
    transfer_rule: Optional[TransferRule] = None,
    rescue_transfer: Optional[RescueTransfer] = None,
    T_grid: Optional[np.ndarray] = None,        # also used as backup grid if transfer_rule is set
    # Boundary handling:
    on_boundary_no_feasible: Literal["prune", "raise"] = "prune",
    # Diagnostics:
    return_H_best: bool = True,
) -> Tuple[Policy, np.ndarray, Optional[np.ndarray]]:
    """
    One improvement pass on a fixed active set.

    Returns:
      u_target : dict {"tau","h","T"} with NaNs off M_target
      M_target : bool mask where at least one feasible control exists
      H_best   : best Hamiltonian value per node (NaN off M_target) if return_H_best else None

    Hamiltonian (deterministic transport form):
      H = flow + kdot * Jk_up + Ldot * JL_up

    NOTE:
      - Gatekeeping uses inward_one_cell_node(active, ...) so it is topology-consistent.
      - If transfer_rule is provided, we still try backup T candidates to avoid false pruning.
      - rescue_transfer, if provided, is only attempted when a candidate fails inwardness.
    """
    J = np.asarray(J, dtype=float)
    omega = np.asarray(omega, dtype=float)
    active = np.asarray(active, dtype=bool)

    if J.shape != grid.shape or omega.shape != grid.shape or active.shape != grid.shape:
        raise ValueError("J, omega, active must all match grid.shape")

    # Derivatives masked against active => stencils crossing inactive nodes are NaN
    Jk_f, Jk_b, JL_f, JL_b = masked_upwind_derivatives(J, active, grid)

    # Backup T candidates (always used; when transfer_rule is None it is the full candidate set)
    T_backup = _default_T_backup(prim, T_grid)

    u_target = empty_policy_like(active)
    M_target = np.zeros_like(active, dtype=bool)
    H_best = np.full_like(J, np.nan) if return_H_best else None

    for node in iter_nodes_where(active):
        i, j = node

        best_H = NEG_INF
        best_u: Optional[Tuple[float, float, float]] = None

        for tau in prim.tau_grid:
            tau = float(tau)
            for h in prim.h_grid:
                h = float(h)

                # ---- Build the T candidate set ----
                T_candidates: list[float] = []

                if transfer_rule is not None:
                    # Analytic proposal
                    T_am = float(transfer_rule(
                        grid, par, prim, s, node, tau, h,
                        J, Jk_f, Jk_b, JL_f, JL_b, omega,
                        eps_drift
                    ))
                    if np.isfinite(T_am):
                        T_candidates.append(float(np.clip(T_am, prim.T_min, prim.T_max)))

                    # Always add backup candidates to prevent false pruning
                    T_candidates.extend([float(t) for t in T_backup])
                    # Deduplicate (order-preserving)
                    seen = set()
                    T_candidates = [t for t in T_candidates if not (t in seen or seen.add(t))]

                else:
                    # No analytic rule: just use the candidate grid
                    T_candidates = [float(t) for t in T_backup]

                # ---- Evaluate candidates ----
                for T in T_candidates:
                    # Static feasibility (must include T-dependent feasibility in your callback)
                    if not static_feasible_node(grid, par, prim, s, node, tau, h, T, omega):
                        continue

                    flow, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega)
                    if not (np.isfinite(flow) and np.isfinite(kdot) and np.isfinite(Ldot)):
                        continue

                    # Inward gate (hard walls + active-set walls)
                    if not inward_one_cell_node(active, node, kdot, Ldot, eps=eps_drift):

                        # Attempt rescue (only if provided)
                        if rescue_transfer is not None:
                            blocked = _blocked_directions(active, node, kdot, Ldot, eps_drift)
                            T_resc = rescue_transfer(grid, par, prim, s, node, tau, h, omega, active, eps_drift, blocked)
                            if T_resc is None or (not np.isfinite(T_resc)):
                                continue
                            T_resc = float(np.clip(T_resc, prim.T_min, prim.T_max))

                            if not static_feasible_node(grid, par, prim, s, node, tau, h, T_resc, omega):
                                continue
                            flow2, kdot2, Ldot2 = node_flow_and_drift(grid, par, prim, s, node, tau, h, T_resc, omega)
                            if not (np.isfinite(flow2) and np.isfinite(kdot2) and np.isfinite(Ldot2)):
                                continue
                            if not inward_one_cell_node(active, node, kdot2, Ldot2, eps=eps_drift):
                                continue

                            # Replace with rescued candidate
                            flow, kdot, Ldot, T = flow2, kdot2, Ldot2, T_resc
                        else:
                            continue

                    # Upwind gradient at this node (epsilon-consistent)
                    Jk = _upwind_scalar_deriv(kdot, Jk_f[i, j], Jk_b[i, j], eps_drift)
                    JL = _upwind_scalar_deriv(Ldot, JL_f[i, j], JL_b[i, j], eps_drift)

                    if not (np.isfinite(Jk) and np.isfinite(JL)):
                        # drift requires a derivative across an inactive or undefined stencil
                        continue

                    H = float(flow + kdot * Jk + Ldot * JL)
                    if H > best_H:
                        best_H = H
                        best_u = (tau, h, T)

        if best_u is None:
            # No feasible control => prune node (no incumbent fallback)
            if on_boundary_no_feasible == "raise" and _is_grid_boundary(node, grid):
                raise RuntimeError(
                    f"No feasible control at boundary node={node} in regime s={s}. "
                    f"This often indicates the computational box is too small or bounds are too tight."
                )
            M_target[node] = False
            continue

        M_target[node] = True
        u_target["tau"][node] = best_u[0]
        u_target["h"][node]   = best_u[1]
        u_target["T"][node]   = best_u[2]
        if H_best is not None:
            H_best[node] = best_H

    u_target = mask_policy(u_target, M_target)
    return u_target, M_target, H_best


def improve_with_prune_closure_legacy_a(
    grid: Grid,
    par: Par,
    prim: Prim,
    s: int,
    J: np.ndarray,
    omega: np.ndarray,
    M: np.ndarray,
    *,
    static_feasible_node: StaticFeasibleNode,
    node_flow_and_drift: NodeFlowAndDrift,
    eps_drift: float = 1e-12,
    transfer_rule: Optional[TransferRule] = None,
    rescue_transfer: Optional[RescueTransfer] = None,
    T_grid: Optional[np.ndarray] = None,
    max_passes: int = 10,
    verbose: bool = False,
    on_boundary_no_feasible: Literal["prune", "raise"] = "prune",
) -> Tuple[Policy, np.ndarray]:
    """
    Repeatedly do:
      (u_targ, M_targ) = argmax on current M_work
      M_new = M_work & M_targ
    until M stops shrinking (or max_passes).

    Returns:
      u_targ masked to M_stable
      M_stable
    """
    if max_passes <= 0:
        raise ValueError("max_passes must be positive.")

    M_work = np.asarray(M, dtype=bool).copy()
    if not np.any(M_work):
        raise RuntimeError("Empty mask entering improve_with_prune_closure (this is not convergence).")

    u_last = empty_policy_like(M_work)

    for p in range(max_passes):
        u_targ, M_targ, _ = policy_improvement_gatekeep(
            grid, par, prim, s, J, omega, M_work,
            static_feasible_node=static_feasible_node,
            node_flow_and_drift=node_flow_and_drift,
            eps_drift=eps_drift,
            transfer_rule=transfer_rule,
            rescue_transfer=rescue_transfer,
            T_grid=T_grid,
            on_boundary_no_feasible=on_boundary_no_feasible,
            return_H_best=False,
        )

        M_new = M_work & M_targ
        if verbose:
            print(f"[prune-closure] s={s} pass={p+1}/{max_passes} |M| {int(M_work.sum())} -> {int(M_new.sum())}")

        u_last = u_targ

        if masks_equal(M_new, M_work):
            return mask_policy(u_last, M_work), M_work

        if not np.any(M_new):
            raise RuntimeError(
                f"Mask collapsed to empty during prune-closure (s={s}). "
                f"This indicates no inward-feasible controls exist on the remaining set."
            )

        M_work = M_new

    return mask_policy(u_last, M_work), M_work


# ============================================================
# OPTIONAL SANITY CHECK (PRINTS OUTPUT)
# - No inward constraints bind because drift=0
# - Hamiltonian reduces to flow, so best tau/h/T maximizes flow.
# ============================================================

def _sanity_check_section_5():
    k = np.linspace(1.0, 2.0, 6)
    L = np.linspace(-0.5, 0.5, 6)
    g = Grid(k, L)

    par = Par(rho=0.05)
    prim = Prim(
        tau_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        h_grid=np.array([0.0, 0.5, 1.0]),
        T_min=-1.0,
        T_max=1.0,
    )

    active = g.interior_mask.copy()
    omega = np.zeros(g.shape)

    # Use a simple J so derivatives exist (but drift=0 so doesn't matter)
    K, LL = np.meshgrid(g.k, g.L, indexing="ij")
    J = 2.0 * K + 3.0 * LL

    # Feasible everywhere
    def static_ok(grid, par, prim, s, node, tau, h, T, omega):
        return True

    # flow is maximized at tau=0.25, h=0.5, T=0.0
    # drift = 0 => inward gate always passes on active set
    def flow_drift(grid, par, prim, s, node, tau, h, T, omega):
        flow = - (tau - 0.25)**2 - (h - 0.5)**2 - (T - 0.0)**2
        return float(flow), 0.0, 0.0

    u_targ, M_targ, H = policy_improvement_gatekeep(
        g, par, prim, s=0, J=J, omega=omega, active=active,
        static_feasible_node=static_ok,
        node_flow_and_drift=flow_drift,
        eps_drift=1e-12,
        transfer_rule=None,
        T_grid=np.array([-1.0, 0.0, 1.0]),
        return_H_best=True,
    )

    print("Section 5 check:")
    print("  |active| =", int(active.sum()), " |M_targ| =", int(M_targ.sum()))
    print("  unique tau:", np.unique(u_targ["tau"][M_targ]))
    print("  unique h  :", np.unique(u_targ["h"][M_targ]))
    print("  unique T  :", np.unique(u_targ["T"][M_targ]))
    print("  max H on active:", float(np.nanmax(H[M_targ])))

# Uncomment to run:
# _sanity_check_section_5()


# In[6]:


# ============================================================
# SECTION 6 — Howard Inner Loop (Plan 2.3 / Plan 4.2)
# Policy Evaluation + Policy Improvement (with prune closure) + Damped Update
# ============================================================

from typing import Optional, Tuple, Dict, Callable
import numpy as np

Policy = Dict[str, np.ndarray]

# Model callback signatures
StaticFeasibleNode = Callable[
    [Grid, Par, Prim, int, Tuple[int, int], float, float, float, np.ndarray],
    bool
]
NodeFlowAndDrift = Callable[
    [Grid, Par, Prim, int, Tuple[int, int], float, float, float, np.ndarray],
    Tuple[float, float, float]  # (flow, kdot, Ldot)
]

# Optional transfer logic signatures (kept very permissive)
TransferRule = Callable[..., float]
RescueTransfer = Callable[..., Optional[float]]


# ============================================================
# ADAPTERS (ONLY NEEDED IF YOU USE THE "Section 3 generator assembly" API)
#   build_masked_system_2_3(...)
#   solve_hjb_on_active(...)
# ============================================================

def build_masked_generator(
    grid: Grid,
    par: Par,
    prim: Prim,
    *,
    s: int,
    u: Policy,
    omega: np.ndarray,
    active: np.ndarray,
    node_flow_and_drift: NodeFlowAndDrift,
    eps_drift: float,
    check_inward: bool,
):
    """
    Adapter so Section 6 can call a "build_masked_generator" even if your
    actual implementation is build_masked_system_2_3 from the generator section.

    Returns:
      A, f, idx_full, inv_full
    """
    omega = np.asarray(omega, dtype=float)

    def node_eval(node: Tuple[int, int]) -> NodeDriftFlow:
        tau = float(u["tau"][node])
        h   = float(u["h"][node])
        T   = float(u["T"][node])
        flow, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega)
        return NodeDriftFlow(kdot=float(kdot), Ldot=float(Ldot), flow=float(flow))

    A, f, act = build_masked_system_2_3(
        grid=grid,
        active=active,
        node_eval=node_eval,
        eps_drift=eps_drift,
        check_inward=check_inward,
    )
    return A, f, act.idx_full, act.inv_full


def solve_hjb_linear_on_active(A, rhs, *, rho: float) -> np.ndarray:
    # (rho I - A)J = rhs
    return solve_hjb_on_active(A, rhs, rho=rho, lam=0.0)


def solve_hjb_poisson_on_active(A, f0, *, rho: float, lam: float, J_couple_act: np.ndarray) -> np.ndarray:
    # ((rho+lam)I - A)J0 = f0 + lam*J1
    rhs = np.asarray(f0, dtype=float) + float(lam) * np.asarray(J_couple_act, dtype=float)
    return solve_hjb_on_active(A, rhs, rho=rho, lam=lam)


def embed_active_to_grid(x_act: np.ndarray, idx_full: np.ndarray, grid: Grid, *, anchor: float = np.nan) -> np.ndarray:
    x_act = np.asarray(x_act, dtype=float).reshape(-1)
    idx_full = np.asarray(idx_full, dtype=np.int64).reshape(-1)
    full = np.full(grid.N, float(anchor), dtype=float)
    full[idx_full] = x_act
    return full.reshape(grid.shape, order="C")


def restrict_grid_to_active(x_grid: np.ndarray, idx_full: np.ndarray) -> np.ndarray:
    x_grid = np.asarray(x_grid, dtype=float)
    idx_full = np.asarray(idx_full, dtype=np.int64).reshape(-1)
    return x_grid.ravel(order="C")[idx_full]


# ============================================================
# SMALL HELPERS
# ============================================================

def _policy_drift_arrays_on_mask(
    grid: Grid,
    par: Par,
    prim: Prim,
    s: int,
    u: Policy,
    omega: np.ndarray,
    mask: np.ndarray,
    *,
    node_flow_and_drift: NodeFlowAndDrift,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (kdot, Ldot) arrays on `mask` using node_flow_and_drift.
    Outside mask, returns zeros.

    Used only for inward gating checks.
    """
    mask = np.asarray(mask, dtype=bool)
    omega = np.asarray(omega, dtype=float)

    kdot = np.zeros(grid.shape, dtype=float)
    Ldot = np.zeros(grid.shape, dtype=float)

    for node in iter_nodes_where(mask):
        tau = float(u["tau"][node])
        h   = float(u["h"][node])
        T   = float(u["T"][node])

        flow, kd, Ld = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega)
        # flow unused here

        kdot[node] = float(kd)
        Ldot[node] = float(Ld)

    return kdot, Ldot


def _select_blend_or_snap_on_mask(
    u_old: Policy,
    u_blend: Policy,
    u_targ: Policy,
    ok: np.ndarray,
    mask: np.ndarray,
) -> Policy:
    """
    Safe snap:
      - on mask: choose blend where ok else target
      - off mask: keep old
    """
    ok = np.asarray(ok, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    choose = ok & mask

    out: Policy = {}
    for key in ("tau", "h", "T"):
        arr = np.asarray(u_old[key], dtype=float).copy()
        arr[mask] = np.where(choose[mask], u_blend[key][mask], u_targ[key][mask])
        out[key] = arr
    return out


# ============================================================
# HOWARD INNER LOOP
# ============================================================

def howard_inner_loop_legacy_a(
    grid: Grid,
    par: Par,
    prim: Prim,
    *,
    lam: float,
    omega1: np.ndarray,
    omega0: np.ndarray,
    J1_init: np.ndarray,
    J0_init: np.ndarray,
    u1_init: Policy,
    u0_init: Policy,
    M1_init: np.ndarray,
    M0_init: np.ndarray,
    static_feasible_node: StaticFeasibleNode,
    node_flow_and_drift: NodeFlowAndDrift,
    # Transfer handling for improvement:
    transfer_rule: Optional[TransferRule] = None,
    rescue_transfer: Optional[RescueTransfer] = None,
    T_grid: Optional[np.ndarray] = None,
    # Howard parameters:
    eta_policy: float = 0.8,
    eps_drift: float = 1e-12,
    m_inner_max: int = 40,
    tol_policy: float = 1e-7,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Policy, Policy, np.ndarray, np.ndarray]:
    """
    Plan 2.3 Howard loop:
      1) Evaluate J1, J0 on current masks using masked generators (active subspace solves)
      2) Improve policies with prune-closure to get (u*_s, M_s_stable)
      3) Damped update: blend on mask, gate, snap to target where blend fails
      4) Stop when policy diff small AND masks stable
    """
    if lam < 0.0:
        raise ValueError("lam must be nonnegative.")
    if not (0.0 <= eta_policy <= 1.0):
        raise ValueError("eta_policy must be in [0,1].")
    if eps_drift < 0.0:
        raise ValueError("eps_drift must be nonnegative.")

    # Copy inputs
    M1 = np.asarray(M1_init, dtype=bool).copy()
    M0 = np.asarray(M0_init, dtype=bool).copy()
    M0 &= M1  # enforce immediate-switch admissibility inside inner loop too

    J1 = np.asarray(J1_init, dtype=float).copy()
    J0 = np.asarray(J0_init, dtype=float).copy()

    u1 = {k: np.asarray(v, dtype=float).copy() for k, v in u1_init.items()}
    u0 = {k: np.asarray(v, dtype=float).copy() for k, v in u0_init.items()}

    # Main Howard iterations
    for m in range(m_inner_max):
        if verbose:
            print(f"\n[Howard] iter {m+1}/{m_inner_max}  |M1|={int(M1.sum())}  |M0|={int(M0.sum())}")

        # =====================================================
        # (A) POLICY EVALUATION
        # =====================================================
        # Regime 1 (absorbing): (rho I - A1) J1 = f1
        A1, f1, idx1, _inv1 = build_masked_generator(
            grid, par, prim, s=1, u=u1, omega=omega1, active=M1,
            node_flow_and_drift=node_flow_and_drift,
            eps_drift=eps_drift,
            check_inward=True,
        )
        J1_act = solve_hjb_linear_on_active(A1, f1, rho=par.rho)
        J1_new = embed_active_to_grid(J1_act, idx1, grid, anchor=np.nan)

        # Regime 0 (Poisson): ((rho+lam) I - A0) J0 = f0 + lam * J1
        A0, f0, idx0, _inv0 = build_masked_generator(
            grid, par, prim, s=0, u=u0, omega=omega0, active=M0,
            node_flow_and_drift=node_flow_and_drift,
            eps_drift=eps_drift,
            check_inward=True,
        )
        J1_on_M0 = restrict_grid_to_active(J1_new, idx0)
        J0_act = solve_hjb_poisson_on_active(A0, f0, rho=par.rho, lam=lam, J_couple_act=J1_on_M0)
        J0_new = embed_active_to_grid(J0_act, idx0, grid, anchor=np.nan)

        # =====================================================
        # (B) POLICY IMPROVEMENT + PRUNE CLOSURE
        # =====================================================
        # NOTE: improve_with_prune_closure must exist elsewhere in your codebase.
        u1_targ, M1_stable = improve_with_prune_closure(
            grid, par, prim, s=1, J=J1_new, omega=omega1, M=M1,
            static_feasible_node=static_feasible_node,
            node_flow_and_drift=node_flow_and_drift,
            eps_drift=eps_drift,
            transfer_rule=transfer_rule,
            rescue_transfer=rescue_transfer,
            T_grid=T_grid,
            max_passes=10,
            verbose=verbose,
        )

        u0_targ, M0_stable = improve_with_prune_closure(
            grid, par, prim, s=0, J=J0_new, omega=omega0, M=M0,
            static_feasible_node=static_feasible_node,
            node_flow_and_drift=node_flow_and_drift,
            eps_drift=eps_drift,
            transfer_rule=transfer_rule,
            rescue_transfer=rescue_transfer,
            T_grid=T_grid,
            max_passes=10,
            verbose=verbose,
        )

        # Enforce cross-regime admissibility
        M0_stable &= M1_stable

        # Mask target policies to stable sets
        u1_targ = mask_policy(u1_targ, M1_stable)
        u0_targ = mask_policy(u0_targ, M0_stable)

        # =====================================================
        # (C) DAMPED UPDATE (BLEND ON MASK) + GATE + SNAP
        # =====================================================
        u1_blend = blend_and_project_on_mask(u1, u1_targ, eta_policy, M1_stable, prim)
        u0_blend = blend_and_project_on_mask(u0, u0_targ, eta_policy, M0_stable, prim)

        # Gate blended policies on stable masks
        kdot1, Ldot1 = _policy_drift_arrays_on_mask(
            grid, par, prim, s=1, u=u1_blend, omega=omega1, mask=M1_stable,
            node_flow_and_drift=node_flow_and_drift
        )
        ok1 = inward_one_cell(M1_stable, kdot1, Ldot1, eps=eps_drift)

        kdot0, Ldot0 = _policy_drift_arrays_on_mask(
            grid, par, prim, s=0, u=u0_blend, omega=omega0, mask=M0_stable,
            node_flow_and_drift=node_flow_and_drift
        )
        ok0 = inward_one_cell(M0_stable, kdot0, Ldot0, eps=eps_drift)

        # Snap where blend fails (on mask); keep old off-mask
        u1_next = _select_blend_or_snap_on_mask(u1, u1_blend, u1_targ, ok1, M1_stable)
        u0_next = _select_blend_or_snap_on_mask(u0, u0_blend, u0_targ, ok0, M0_stable)

        # Final masking (keeps NaNs off-mask)
        u1_next = mask_policy(u1_next, M1_stable)
        u0_next = mask_policy(u0_next, M0_stable)

        # Defensive checks (FIXED): check inwardness ONLY ON THE MASK
        kdot1F, Ldot1F = _policy_drift_arrays_on_mask(
            grid, par, prim, s=1, u=u1_next, omega=omega1, mask=M1_stable,
            node_flow_and_drift=node_flow_and_drift
        )
        ok1F = inward_one_cell(M1_stable, kdot1F, Ldot1F, eps=eps_drift)
        if not ok1F[M1_stable].all():
            raise RuntimeError("Post-snap u1 violates inwardness on M1_stable (should be impossible).")

        kdot0F, Ldot0F = _policy_drift_arrays_on_mask(
            grid, par, prim, s=0, u=u0_next, omega=omega0, mask=M0_stable,
            node_flow_and_drift=node_flow_and_drift
        )
        ok0F = inward_one_cell(M0_stable, kdot0F, Ldot0F, eps=eps_drift)
        if not ok0F[M0_stable].all():
            raise RuntimeError("Post-snap u0 violates inwardness on M0_stable (should be impossible).")

        # =====================================================
        # (D) STOPPING
        # =====================================================
        pol_diff = policy_supnorm(u1_next, u1, M1_stable) + policy_supnorm(u0_next, u0, M0_stable)
        mask_same = masks_equal(M1_stable, M1) and masks_equal(M0_stable, M0)

        if verbose:
            print(f"[Howard] pol_diff={pol_diff:.3e}  masks_same={mask_same}")

        if pol_diff < tol_policy and mask_same:
            return J1_new, J0_new, u1_next, u0_next, M1_stable, M0_stable

        # Carry forward
        J1, J0 = J1_new, J0_new
        u1, u0 = u1_next, u0_next
        M1, M0 = M1_stable, (M0_stable & M1_stable)

    # Max iters reached: return last iterates
    return J1, J0, u1, u0, M1, M0


# In[7]:


# ============================================================
# SECTION 7 (FIXED) — Outer Loop Driver (Plan 2.3 / Plan 4.2)
#
# Fixes vs your current Section 7:
#   1) Owner-effective domain: Momega_eff_s = Momega_s ∩ M_s
#      so update_private_omega never sees NaN planner policies.
#   2) Convergence norms computed on stable CORES:
#        core_s = M_s_old ∩ M_s_new
#      so shrinking masks can’t fake convergence.
#   3) Mask stability uses SET equality (not just size).
#   4) Ex-post "full peel" truly starts from base_active (can expand vs warm peel),
#      and regime-0 peel is computed inside V1_full (immediate-switch admissibility).
#   5) Optional final re-solve if full peel changes the domain, with safe initialization
#      of (J,u) on any newly-added nodes.
# ============================================================

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np

Policy = Dict[str, np.ndarray]

UpdatePrivateOmega = Callable[[Grid, Par, Prim, int, np.ndarray, Policy, np.ndarray], np.ndarray]
PrimitiveFeasibleSet = Callable[[Grid, Par, Prim], np.ndarray]


def _drift_node_from_flow_and_drift(node_flow_and_drift: NodeFlowAndDrift):
    """Adapter: returns (kdot, Ldot) using the same primitive callback."""
    def drift_node(grid: Grid, par: Par, prim: Prim, s: int, node: tuple[int, int],
                   tau: float, h: float, T: float, omega: np.ndarray) -> tuple[float, float]:
        _, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega)
        return float(kdot), float(Ldot)
    return drift_node


def _assert_policy_finite_on_mask(u: Policy, mask: np.ndarray, *, name: str) -> None:
    """Fail fast if u has NaN/Inf on the mask that a downstream solver will use."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise RuntimeError(f"{name}: required mask is empty.")
    for key in ("tau", "h", "T"):
        if key not in u:
            raise KeyError(f"{name}: policy missing key '{key}'")
        arr = np.asarray(u[key], dtype=float)
        if arr.shape != mask.shape:
            raise ValueError(f"{name}: policy '{key}' has wrong shape.")
        if not np.isfinite(arr[mask]).all():
            raise RuntimeError(f"{name}: policy '{key}' is not finite on required mask.")


def _fill_policy_defaults_on_mask(u: Policy, mask: np.ndarray, prim: Prim) -> Policy:
    """
    Ensure policy is finite on `mask` by filling NaNs/Infs with safe defaults.
    This is used only for the optional final re-solve when full peel *expands* the domain.
    """
    mask = np.asarray(mask, dtype=bool)
    out = {k: np.asarray(v, dtype=float).copy() for k, v in u.items()}

    tau0 = float(np.median(prim.tau_grid))
    h0   = float(np.median(prim.h_grid))
    T0   = float(0.5 * (prim.T_min + prim.T_max))

    # Clip defaults to bounds
    tau0 = float(np.clip(tau0, prim.tau_min, prim.tau_max))
    h0   = float(np.clip(h0, prim.h_min, prim.h_max))
    T0   = float(np.clip(T0, prim.T_min, prim.T_max))

    for key, default in (("tau", tau0), ("h", h0), ("T", T0)):
        if key not in out:
            raise KeyError(f"Policy missing key '{key}'")
        arr = out[key]
        if arr.shape != mask.shape:
            raise ValueError("Policy arrays and mask must have the same shape.")
        bad = mask & (~np.isfinite(arr))
        if np.any(bad):
            arr[bad] = default
        out[key] = arr

    return out


def _fill_J_defaults_on_mask(J: np.ndarray, mask: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    """Ensure J is finite on mask by filling NaNs/Infs with a default."""
    J = np.asarray(J, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    bad = mask & (~np.isfinite(J))
    if np.any(bad):
        J[bad] = float(default)
    return J


def _core_max_norm(delta: np.ndarray, core: np.ndarray, *, label: str) -> float:
    core = np.asarray(core, dtype=bool)
    if not np.any(core):
        raise RuntimeError(f"Empty core mask in {label} (mask moved/collapsed).")
    return float(max_norm_on_mask(delta, core))


def outer_loop_solver_legacy_a(
    grid: Grid,
    par: Par,
    prim: Prim,
    *,
    lam: float,

    # initial guesses:
    omega1_init: np.ndarray,
    omega0_init: np.ndarray,
    J1_init: np.ndarray,
    J0_init: np.ndarray,
    u1_init: Policy,
    u0_init: Policy,

    # owner-domain masks:
    Momega1: np.ndarray,
    Momega0: np.ndarray,

    # model callbacks:
    primitive_feasible_set: PrimitiveFeasibleSet,
    update_private_omega: UpdatePrivateOmega,
    static_feasible_node: StaticFeasibleNode,
    node_flow_and_drift: NodeFlowAndDrift,

    # outer controls:
    zeta_omega: float = 0.2,
    N_peel: int = 5,
    eps_drift: float = 1e-12,
    max_outer: int = 200,
    tol_outer: float = 1e-6,

    # stability controls:
    stable_window: int = 3,        # require this many consecutive identical masks
    min_mask_size: int = 10,       # hard fail if masks collapse below this size

    # howard controls:
    eta_policy: float = 0.8,
    m_inner_max: int = 40,
    tol_policy: float = 1e-7,
    transfer_rule: Optional[TransferRule] = None,
    rescue_transfer: Optional[RescueTransfer] = None,
    T_grid: Optional[np.ndarray] = None,

    # ex-post:
    do_full_peel: bool = True,
    peel_full_max: int = 500,
    resolve_after_full_peel: bool = True,

    verbose: bool = False,
) -> Dict[str, object]:
    """
    Full outer loop:
      1) update ω on owner-effective domain (Momega ∩ M), under-relax
      2) quarantine ω outside that domain (nearest-neighbour fill)
      3) warm viability peel
      4) Howard inner loop on frozen ω
      5) convergence on core norms + mask set-stability
      6) optional ex-post peel-to-fixpoint (true kernel check)
      7) optional final re-solve if full peel changes the domain
    """
    if lam < 0.0:
        raise ValueError("lam must be nonnegative.")
    if not (0.0 < zeta_omega <= 1.0):
        raise ValueError("zeta_omega must be in (0,1].")
    if not (0.0 < eta_policy <= 1.0):
        raise ValueError("eta_policy must be in (0,1].")
    if max_outer <= 0:
        raise ValueError("max_outer must be positive.")
    if stable_window <= 0:
        raise ValueError("stable_window must be positive.")

    # ----------------------------------------------------
    # Base feasible set S and truncation interior B°
    # ----------------------------------------------------
    S = np.asarray(primitive_feasible_set(grid, par, prim), dtype=bool)
    if S.shape != grid.shape:
        raise ValueError("primitive_feasible_set must return mask with grid.shape")
    B = np.asarray(grid.interior_mask, dtype=bool)
    base_active = S & B

    # Start viability at base_active; enforce immediate-switch admissibility
    V1 = base_active.copy()
    V0 = base_active.copy()
    V0 &= V1

    # Initial evaluation masks
    M1 = V1.copy()
    M0 = V0.copy()

    # State
    omega1 = np.asarray(omega1_init, dtype=float).copy()
    omega0 = np.asarray(omega0_init, dtype=float).copy()
    J1 = np.asarray(J1_init, dtype=float).copy()
    J0 = np.asarray(J0_init, dtype=float).copy()
    u1 = {k: np.asarray(v, dtype=float).copy() for k, v in u1_init.items()}
    u0 = {k: np.asarray(v, dtype=float).copy() for k, v in u0_init.items()}

    Momega1 = np.asarray(Momega1, dtype=bool).copy()
    Momega0 = np.asarray(Momega0, dtype=bool).copy()
    if not np.any(Momega1) or not np.any(Momega0):
        raise ValueError("Momega masks must contain at least one True entry for quarantine_fill_nearest.")

    drift_node = _drift_node_from_flow_and_drift(node_flow_and_drift)

    history: list[dict] = []

    # mask set-stability tracking
    M1_prev: Optional[np.ndarray] = None
    M0_prev: Optional[np.ndarray] = None
    stable_count = 0

    # owner-effective domain tracking for ω core norms
    Momega1_eff_prev: Optional[np.ndarray] = None
    Momega0_eff_prev: Optional[np.ndarray] = None

    for n in range(max_outer):
        if verbose:
            print(f"\n[Outer] iter {n+1}/{max_outer} |M1|={int(M1.sum())} |M0|={int(M0.sum())}")

        # ----------------------------------------------------
        # (1) Update ω on owner-effective domain + under-relax
        # ----------------------------------------------------
        Momega1_eff = Momega1 & M1
        Momega0_eff = Momega0 & M0
        if not np.any(Momega1_eff) or not np.any(Momega0_eff):
            raise RuntimeError(
                "Owner-effective domain is empty (Momega_s ∩ M_s). "
                "Either the planner mask collapsed, or Momega_s doesn't overlap M_s."
            )

        _assert_policy_finite_on_mask(u1, Momega1_eff, name="u1 (owner-effective)")
        _assert_policy_finite_on_mask(u0, Momega0_eff, name="u0 (owner-effective)")

        omega1_new = np.asarray(update_private_omega(grid, par, prim, 1, omega1, u1, Momega1_eff), dtype=float)
        omega0_new = np.asarray(update_private_omega(grid, par, prim, 0, omega0, u0, Momega0_eff), dtype=float)
        if omega1_new.shape != grid.shape or omega0_new.shape != grid.shape:
            raise ValueError("update_private_omega must return full-grid arrays with grid.shape")

        omega1_half = (1.0 - zeta_omega) * omega1 + zeta_omega * omega1_new
        omega0_half = (1.0 - zeta_omega) * omega0 + zeta_omega * omega0_new

        # Require finite ω on the owner-effective domain before fill
        if not np.isfinite(omega1_half[Momega1_eff]).all():
            raise RuntimeError("Non-finite omega1 on owner-effective mask after under-relaxation.")
        if not np.isfinite(omega0_half[Momega0_eff]).all():
            raise RuntimeError("Non-finite omega0 on owner-effective mask after under-relaxation.")

        # Quarantine outside owner-effective domain: nearest neighbour fill
        omega1_ext = quarantine_fill_nearest(omega1_half, Momega1_eff)
        omega0_ext = quarantine_fill_nearest(omega0_half, Momega0_eff)

        # ----------------------------------------------------
        # (2) Warm viability peel
        # ----------------------------------------------------
        V1 = viability_peel_warm(
            V1, base_active, grid, par, prim, omega1_ext, s=1,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            N_peel=N_peel, eps_drift=eps_drift,
            verbose=False
        )
        V0 = viability_peel_warm(
            V0, base_active, grid, par, prim, omega0_ext, s=0,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            N_peel=N_peel, eps_drift=eps_drift,
            verbose=False
        )
        V0 &= V1  # immediate-switch admissibility

        M1 = V1.copy()
        M0 = V0.copy()

        if int(M1.sum()) < min_mask_size or int(M0.sum()) < min_mask_size:
            raise RuntimeError(
                f"Mask collapse after warm peel: |M1|={int(M1.sum())}, |M0|={int(M0.sum())}. "
                "This is not convergence."
            )

        # ----------------------------------------------------
        # (3) Howard inner loop on frozen ω_ext
        # ----------------------------------------------------
        J1_new, J0_new, u1_new, u0_new, M1_new, M0_new = howard_inner_loop(
            grid, par, prim,
            lam=lam,
            omega1=omega1_ext, omega0=omega0_ext,
            J1_init=J1, J0_init=J0,
            u1_init=u1, u0_init=u0,
            M1_init=M1, M0_init=M0,
            static_feasible_node=static_feasible_node,
            node_flow_and_drift=node_flow_and_drift,
            transfer_rule=transfer_rule,
            rescue_transfer=rescue_transfer,
            T_grid=T_grid,
            eta_policy=eta_policy,
            eps_drift=eps_drift,
            m_inner_max=m_inner_max,
            tol_policy=tol_policy,
            verbose=verbose,
        )

        if int(M1_new.sum()) < min_mask_size or int(M0_new.sum()) < min_mask_size:
            raise RuntimeError(
                f"Mask collapse after inner loop: |M1_new|={int(M1_new.sum())}, |M0_new|={int(M0_new.sum())}. "
                "This is not convergence."
            )

        # ----------------------------------------------------
        # (4) Convergence checks on stable cores + mask set-stability
        # ----------------------------------------------------
        core1 = M1 & M1_new
        core0 = M0 & M0_new

        # ω cores: compare on intersection of owner-effective domains
        core_om1 = Momega1_eff if (Momega1_eff_prev is None) else (Momega1_eff & Momega1_eff_prev)
        core_om0 = Momega0_eff if (Momega0_eff_prev is None) else (Momega0_eff & Momega0_eff_prev)

        d_omega = _core_max_norm(omega1_ext - omega1, core_om1, label="d_omega1") + \
                  _core_max_norm(omega0_ext - omega0, core_om0, label="d_omega0")
        d_J = _core_max_norm(J1_new - J1, core1, label="d_J1") + \
              _core_max_norm(J0_new - J0, core0, label="d_J0")
        d_u = policy_supnorm(u1_new, u1, core1) + policy_supnorm(u0_new, u0, core0)

        history.append({
            "outer_iter": n,
            "d_omega": float(d_omega),
            "d_J": float(d_J),
            "d_u": float(d_u),
            "size_M1": int(M1_new.sum()),
            "size_M0": int(M0_new.sum()),
            "size_core1": int(core1.sum()),
            "size_core0": int(core0.sum()),
        })

        if verbose:
            print(f"[Outer] d_omega={d_omega:.3e}  d_J={d_J:.3e}  d_u={d_u:.3e}  "
                  f"|M1|={int(M1_new.sum())} |M0|={int(M0_new.sum())}")

        # mask set stability (inner-loop masks)
        if (M1_prev is not None) and (M0_prev is not None) and masks_equal(M1_new, M1_prev) and masks_equal(M0_new, M0_prev):
            stable_count += 1
        else:
            stable_count = 0
        M1_prev = M1_new.copy()
        M0_prev = M0_new.copy()

        # Commit state (after computing diffs)
        omega1, omega0 = omega1_ext, omega0_ext
        J1, J0 = J1_new, J0_new
        u1, u0 = u1_new, u0_new

        Momega1_eff_prev = Momega1_eff.copy()
        Momega0_eff_prev = Momega0_eff.copy()

        # Keep viability consistent with inner pruning
        V1 &= M1_new
        V0 &= (M0_new & V1)
        M1, M0 = V1.copy(), V0.copy()

        # Convergence condition
        if (d_omega < tol_outer) and (d_J < tol_outer) and (d_u < tol_outer) and (stable_count >= stable_window):
            if verbose:
                print("[Outer] Converged.")
            break

    # ----------------------------------------------------
    # (5) Ex-post full peel-to-fixpoint (optional true-kernel check)
    # ----------------------------------------------------
    if do_full_peel:
        # Regime 1 true kernel on base_active
        V1_full = viability_peel_to_fixpoint(
            base_active.copy(), base_active, grid, par, prim, omega1, s=1,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            eps_drift=eps_drift,
            max_iter=peel_full_max,
            verbose=False
        )

        # Regime 0 kernel computed INSIDE V1_full
        base0 = base_active & V1_full
        V0_full = viability_peel_to_fixpoint(
            base0.copy(), base0, grid, par, prim, omega0, s=0,
            static_feasible_node=static_feasible_node,
            drift_node=drift_node,
            eps_drift=eps_drift,
            max_iter=peel_full_max,
            verbose=False
        )
        V0_full &= V1_full
    else:
        V1_full, V0_full = V1.copy(), V0.copy()

    # ----------------------------------------------------
    # (6) Optional final re-solve if full peel changed the domain
    # ----------------------------------------------------
    if resolve_after_full_peel:
        M1_final = V1_full.copy()
        M0_final = V0_full.copy()

        if (not masks_equal(M1_final, M1)) or (not masks_equal(M0_final, M0)):
            if verbose:
                print("[Outer] Full peel changed domain -> final re-solve on frozen ω and fixed masks.")

            # If full peel expanded the domain, fill (J,u) safely on newly active nodes
            u1_fill = _fill_policy_defaults_on_mask(u1, M1_final, prim)
            u0_fill = _fill_policy_defaults_on_mask(u0, M0_final, prim)
            J1_fill = _fill_J_defaults_on_mask(J1, M1_final, default=0.0)
            J0_fill = _fill_J_defaults_on_mask(J0, M0_final, default=0.0)

            J1_r, J0_r, u1_r, u0_r, M1_r, M0_r = howard_inner_loop(
                grid, par, prim,
                lam=lam,
                omega1=omega1, omega0=omega0,
                J1_init=J1_fill, J0_init=J0_fill,
                u1_init=u1_fill, u0_init=u0_fill,
                M1_init=M1_final, M0_init=M0_final,
                static_feasible_node=static_feasible_node,
                node_flow_and_drift=node_flow_and_drift,
                transfer_rule=transfer_rule,
                rescue_transfer=rescue_transfer,
                T_grid=T_grid,
                eta_policy=eta_policy,
                eps_drift=eps_drift,
                m_inner_max=m_inner_max,
                tol_policy=tol_policy,
                verbose=verbose,
            )

            # Keep returned kernels consistent with any inner-loop pruning
            V1_full &= M1_r
            V0_full &= (M0_r & V1_full)

            J1, J0 = J1_r, J0_r
            u1, u0 = u1_r, u0_r

    return {
        "omega1": omega1, "omega0": omega0,
        "J1": J1, "J0": J0,
        "u1": u1, "u0": u0,
        "V1": V1_full, "V0": V0_full,
        "M1": V1_full.copy(), "M0": V0_full.copy(),
        "history": history,
    }


# In[8]:


# ============================================================
# TEST CELL — Plan 2.3 / Plan 4.2 Smoke Tests (Sections 1–7)
# ============================================================

import numpy as np

def _require(names):
    missing = [n for n in names if n not in globals()]
    if missing:
        raise NameError(
            "Missing definitions in this kernel:\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun the earlier sections that define these (especially Section 1)."
        )

def _bind_smoke_compat_names():
    """Bind legacy section-8 API names expected by smoke tests if needed."""
    alias_pairs = [
        ("empty_policy_like", "empty_policy_like_legacy_a"),
        ("mask_policy", "mask_policy_legacy_a"),
        ("policy_improvement_gatekeep", "policy_improvement_gatekeep_legacy_a"),
        ("improve_with_prune_closure", "improve_with_prune_closure_legacy_a"),
        ("howard_inner_loop", "howard_inner_loop_legacy_a"),
        ("outer_loop_solver", "outer_loop_solver_legacy_a"),
    ]
    g = globals()
    for public_name, legacy_name in alias_pairs:
        if public_name not in g and legacy_name in g:
            g[public_name] = g[legacy_name]


def run_plan23_smoke_tests(verbose=True):
    _bind_smoke_compat_names()
    # --------- check dependencies ----------
    _require([
        # Section 1
        "Grid", "Par", "Prim",
        "empty_policy_like", "mask_policy",
        "inward_one_cell", "inward_one_cell_node",
        "quarantine_fill_nearest",
        "blend_and_project_on_mask", "policy_supnorm",
        "max_norm_on_mask", "masks_stable",
        # Section 2
        "masked_upwind_derivatives",
        # Section 3
        "viability_peel_warm", "viability_peel_to_fixpoint",
        # Section 4
        "build_masked_generator",
        "solve_hjb_linear_on_active", "solve_hjb_poisson_on_active",
        "embed_active_to_grid", "restrict_grid_to_active",
        # Section 5
        "policy_improvement_gatekeep", "improve_with_prune_closure",
        # Section 6
        "howard_inner_loop",
        # Section 7
        "outer_loop_solver",
    ])

    print("=== Plan 2.3 Smoke Tests ===")

    # --------- set up small toy problem ----------
    k = np.linspace(1.0, 2.0, 6)
    L = np.linspace(-0.5, 0.5, 6)
    grid = Grid(k, L)
    par = Par(rho=0.2)
    prim = Prim(
        tau_grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        h_grid=np.array([0.0, 0.5, 1.0]),
        T_min=-1.0, T_max=1.0
    )

    # masks
    M = grid.interior_mask.copy()
    Momega1 = M.copy()
    Momega0 = M.copy()

    omega1_init = np.zeros(grid.shape)
    omega0_init = np.zeros(grid.shape)
    J1_init = np.zeros(grid.shape)
    J0_init = np.zeros(grid.shape)

    u1 = empty_policy_like(M)
    u0 = empty_policy_like(M)
    for u in (u1, u0):
        u["tau"][M] = 0.5
        u["h"][M]   = 0.0
        u["T"][M]   = 0.0
    u1 = mask_policy(u1, M)
    u0 = mask_policy(u0, M)

    # --------- toy callbacks ----------
    def primitive_feasible_set(grid, par, prim):
        # everything feasible (on the full grid); outer driver intersects with interior anyway
        return np.ones(grid.shape, dtype=bool)

    def update_private_omega(grid, par, prim, s, omega_old, u_s, Momega_s):
        # constant omega for toy test
        return omega_old

    def static_ok(grid, par, prim, s, node, tau, h, T, omega):
        # always feasible in toy test
        return True

    # concave flow with unique maximizer at (tau=0.25, h=0.5, T=0.0); drift = 0
    def flow_and_drift(grid, par, prim, s, node, tau, h, T, omega):
        flow = - (tau - 0.25)**2 - (h - 0.5)**2 - (T - 0.0)**2
        return float(flow), 0.0, 0.0

    # ==========================================================
    # TEST 1: inward_one_cell semantics (mask-restricted check)
    # ==========================================================
    try:
        kdot = np.zeros(grid.shape)
        Ldot = np.zeros(grid.shape)
        ok = inward_one_cell(M, kdot, Ldot, eps=1e-12)

        # Correct check is ON mask:
        assert ok[M].all()

        # This is the common pitfall; it will usually be False:
        pitfall = ok.all()

        print("TEST 1 PASS: inward_one_cell is True on-mask; ok.all() pitfall =", pitfall)
    except Exception as e:
        print("TEST 1 FAIL:", repr(e))
        raise

    # ==========================================================
    # TEST 2: masked generator evaluation (zero drift => A=0)
    # ==========================================================
    try:
        A, f, idx, inv = build_masked_generator(
            grid, par, prim, s=1, u=u1, omega=omega1_init, active=M,
            node_flow_and_drift=lambda grid, par, prim, s, node, tau, h, T, omega: (1.0, 0.0, 0.0),
            eps_drift=1e-12,
            check_inward=True
        )
        J_act = solve_hjb_linear_on_active(A, f, rho=par.rho)
        J_full = embed_active_to_grid(J_act, idx, grid, anchor=np.nan)
        target = 1.0 / par.rho
        err = np.nanmax(np.abs(J_full[M] - target))
        assert err < 1e-10
        print(f"TEST 2 PASS: generator solve (A=0) err={err:.3e}")
    except Exception as e:
        print("TEST 2 FAIL:", repr(e))
        raise

    # ==========================================================
    # TEST 3: policy improvement chooses unique maximizer
    # ==========================================================
    try:
        # any J works (drift=0), but keep finite
        K, LL = np.meshgrid(grid.k, grid.L, indexing="ij")
        J = 2.0 * K + 3.0 * LL

        u_targ, M_targ, H = policy_improvement_gatekeep(
            grid, par, prim, s=0, J=J, omega=omega0_init, active=M,
            static_feasible_node=static_ok,
            node_flow_and_drift=flow_and_drift,
            eps_drift=1e-12,
            transfer_rule=None,
            T_grid=np.array([-1.0, 0.0, 1.0]),
            return_H_best=True,
        )
        assert int(M_targ.sum()) == int(M.sum())
        assert np.allclose(np.unique(u_targ["tau"][M_targ]), 0.25)
        assert np.allclose(np.unique(u_targ["h"][M_targ]), 0.5)
        assert np.allclose(np.unique(u_targ["T"][M_targ]), 0.0)
        print("TEST 3 PASS: policy improvement picks (0.25,0.5,0.0)")
    except Exception as e:
        print("TEST 3 FAIL:", repr(e))
        raise

    # ==========================================================
    # TEST 4: Howard inner loop runs and converges to same policy
    # ==========================================================
    try:
        J1, J0, u1_out, u0_out, M1_out, M0_out = howard_inner_loop(
            grid, par, prim,
            lam=0.3,
            omega1=omega1_init, omega0=omega0_init,
            J1_init=J1_init, J0_init=J0_init,
            u1_init=u1, u0_init=u0,
            M1_init=M.copy(), M0_init=M.copy(),
            static_feasible_node=static_ok,
            node_flow_and_drift=flow_and_drift,
            T_grid=np.array([-1.0, 0.0, 1.0]),
            eta_policy=0.8,
            eps_drift=1e-12,
            m_inner_max=10,
            tol_policy=1e-12,
            verbose=False,
        )
        assert np.allclose(np.unique(u1_out["tau"][M1_out]), 0.25)
        assert np.allclose(np.unique(u1_out["h"][M1_out]), 0.5)
        assert np.allclose(np.unique(u1_out["T"][M1_out]), 0.0)
        print("TEST 4 PASS: Howard loop converges to correct policy")
    except Exception as e:
        print("TEST 4 FAIL:", repr(e))
        raise

    # ==========================================================
    # TEST 5: Outer loop driver runs end-to-end
    # ==========================================================
    try:
        sol = outer_loop_solver(
            grid, par, prim,
            lam=0.3,
            omega1_init=omega1_init, omega0_init=omega0_init,
            J1_init=J1_init, J0_init=J0_init,
            u1_init=u1, u0_init=u0,
            Momega1=Momega1, Momega0=Momega0,
            primitive_feasible_set=primitive_feasible_set,
            update_private_omega=update_private_omega,
            static_feasible_node=static_ok,
            node_flow_and_drift=flow_and_drift,
            zeta_omega=0.5,
            N_peel=2,
            max_outer=5,
            tol_outer=1e-12,
            eta_policy=0.8,
            m_inner_max=10,
            tol_policy=1e-12,
            T_grid=np.array([-1.0, 0.0, 1.0]),
            do_full_peel=False,
            verbose=False,
        )
        M1_out = sol["M1"]
        assert np.allclose(np.unique(sol["u1"]["tau"][M1_out]), 0.25)
        assert np.allclose(np.unique(sol["u1"]["h"][M1_out]), 0.5)
        assert np.allclose(np.unique(sol["u1"]["T"][M1_out]), 0.0)
        print(f"TEST 5 PASS: outer loop ran ({len(sol['history'])} outer iters)")
    except Exception as e:
        print("TEST 5 FAIL:", repr(e))
        raise

    print("✅ ALL TESTS PASSED")


if __name__ == "__main__":
    # Optional smoke tests / kernel diagnostics
    # Run them:
    run_plan23_smoke_tests()


# In[9]:


if __name__ == "__main__":
    for name in ["viability_peel_warm", "viability_peel_step", "viability_peel_to_fixpoint"]:
        print(name, name in globals())


# In[10]:


from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Small numeric guards
_EPS_I = 1e-14   # for I in (0,1)
_EPS_W = 1e-14   # for wealth denominator k+L
_EPS_K = 1e-14   # for k>0 checks (optional)


@dataclass(frozen=True)
class Par:
    """
    Structural parameters for Track B Markov-perfect Ramsey planner
    with Poisson automation and two assets.
    """
    rho: float
    gamma: float
    chi: float
    delta: float
    g: float
    sigma: float
    I0: float
    I1: float

    def I(self, s: int) -> float:
        """Regime-dependent automation frontier."""
        return self.I1 if int(s) == 1 else self.I0


def Phi(I: float, *, eps: float = _EPS_I) -> float:
    """
    Production-function constant:
        Φ(I) = I^{-I} (1-I)^{-(1-I)}.

    Implemented in log form for numerical stability:
        log Φ(I) = -I log I - (1-I) log(1-I).
    """
    I = float(np.clip(float(I), eps, 1.0 - eps))
    logPhi = -I * np.log(I) - (1.0 - I) * np.log(1.0 - I)
    return float(np.exp(logPhi))


def production_block(k: float, s: int, par: Par) -> tuple[float, float, float]:
    """
    Node-local production/pricing block.

        Y_s(k)    = Φ(I_s) k^{I_s}
        w_s(k)    = (1-I_s) Y_s(k)
        R^K_s(k)  = I_s Y_s(k)/k
        r^k_s(k)  = R^K_s(k) - δ
    """
    k = float(k)
    I_s = float(par.I(s))

    if not np.isfinite(k):
        return float("nan"), float("nan"), float("nan")

    if k <= 0.0:
        # Defensive fallback: avoid invalid powers/division.
        return 0.0, 0.0, -float(par.delta)

    Y = Phi(I_s) * (k ** I_s)
    w = (1.0 - I_s) * Y
    R_K = I_s * (Y / k)
    r_k = R_K - par.delta
    return float(Y), float(w), float(r_k)


def market_clearing_risky_share(
    k: float,
    L: float,
    h: float,
    *,
    eps_w: float = _EPS_W,
) -> float:
    """
    Market-clearing risky share for private owners:
        W^K = k + L
        π^{mc} = (k - h)/(k + L)

    Fixes: avoids ZeroDivisionError when k+L == 0 (or nearly 0).
    """
    k = float(k)
    L = float(L)
    h = float(h)

    W = k + L
    if (not np.isfinite(W)) or (W <= eps_w):
        # Returning NaN makes infeasibilities obvious; gate these out upstream.
        return float("nan")

    return float((k - h) / W)


def safe_rate(
    k: float,
    L: float,
    h: float,
    tau: float,
    r_k: float,
    par: Par,
    *,
    eps_w: float = _EPS_W,
    clamp_tau: bool = True,
) -> float:
    """
    Endogenous *pre-tax* short rate schedule (NDL-only, unconstrained Merton pin-down):
        r_f = r_k - γ(1-τ) σ^2 π^{mc},
    where π^{mc} = (k-h)/(k+L).

    Fixes:
      - robust handling of k+L <= 0 (returns NaN rather than dividing by zero)
      - optional clipping of tau into [0,1] to prevent accidental overshoots
    """
    k = float(k)
    L = float(L)
    h = float(h)
    tau = float(tau)
    r_k = float(r_k)

    if not (np.isfinite(k) and np.isfinite(L) and np.isfinite(h) and np.isfinite(tau) and np.isfinite(r_k)):
        return float("nan")

    if clamp_tau:
        # Allows mild numerical overshoots without exploding (1 - tau).
        tau = float(np.clip(tau, 0.0, 1.0))

    pi_mc = market_clearing_risky_share(k, L, h, eps_w=eps_w)
    if not np.isfinite(pi_mc):
        return float("nan")

    return float(r_k - par.gamma * (1.0 - tau) * (par.sigma ** 2) * pi_mc)


# In[11]:


import numpy as np
from typing import Any, Optional, Tuple

def static_feasible_node(
    grid: Any,
    par: Any,
    prim: Any,
    s: int,
    node: Tuple[int, int],
    tau: float,
    h: float,
    *args,
    eps_c: float = 1e-8,
) -> bool:
    """
    Nodewise static feasibility check (hard solvency + consumption positivity).

    Supports BOTH call patterns:
      (A) static_feasible_node(..., tau, h, T, omega)
      (B) static_feasible_node(..., tau, h, omega)

    If T is omitted, we check whether there exists some T ∈ [T_min, T_max]
    that can make worker consumption strictly > eps_c at this node.
    """

    # -------------------------
    # Parse args: (T, omega) or (omega,)
    # -------------------------
    T: Optional[float]
    omega: np.ndarray

    if len(args) == 2:
        T = float(args[0])
        omega = np.asarray(args[1])
    elif len(args) == 1:
        T = None
        omega = np.asarray(args[0])
    else:
        raise TypeError(
            "static_feasible_node expects either (..., tau, h, T, omega) "
            "or (..., tau, h, omega)."
        )

    # omega must be a 2D array for omega[i, j]
    if omega.ndim != 2:
        return False

    i, j = node

    # Defensive index bounds (avoid hard crashes in loops)
    try:
        k = float(grid.k[i])
        L = float(grid.L[j])
    except Exception:
        return False

    # -------------------------
    # 0) Basic numeric sanity
    # -------------------------
    if not (np.isfinite(k) and np.isfinite(L) and np.isfinite(tau) and np.isfinite(h)):
        return False

    # -------------------------
    # 1) State space bounds (hard solvency / no-default primitives)
    # -------------------------
    W = k + L
    if W <= 0.0:
        return False
    if L < -k:
        return False

    # -------------------------
    # 2) Primitive instrument bounds
    # -------------------------
    tau_min = float(getattr(prim, "tau_min", -np.inf))
    tau_max = float(getattr(prim, "tau_max",  np.inf))
    if tau < tau_min or tau > tau_max:
        return False

    # -------------------------
    # 3) Control bounds dependent on state: H ∈ [max(0,-L), k]
    # -------------------------
    h_lo = max(0.0, -L)
    h_hi = k
    if h < h_lo or h > h_hi:
        return False

    # -------------------------
    # 4) Owner consumption positivity (always check)
    #    C_K = omega(i,j) * (k+L)
    # -------------------------
    try:
        omg = float(omega[i, j])
    except Exception:
        return False

    if (not np.isfinite(omg)) or (omg <= 0.0):
        return False

    C_K = omg * W
    if C_K <= eps_c:
        return False

    # -------------------------
    # 5) Worker consumption positivity
    #    C_W = w(k,s) + T
    # -------------------------
    _, w, _ = production_block(k, s, par)
    if not np.isfinite(w):
        return False

    T_min = float(getattr(prim, "T_min", -np.inf))
    T_max = float(getattr(prim, "T_max",  np.inf))
    if T_max < T_min:
        return False

    if T is None:
        # Existence check: since C_W increases in T, best case is T = T_max.
        # Need STRICT positivity: w + T_max > eps_c.
        if float(w) + T_max <= eps_c:
            return False
    else:
        if not np.isfinite(T):
            return False
        if T < T_min or T > T_max:
            return False

        C_W = float(w) + float(T)
        if C_W <= eps_c:
            return False

    return True


# In[12]:


import numpy as np
from typing import Any, Tuple

def crra_utility(c: float, gamma: float, *, eps: float = 1e-14) -> float:
    """
    CRRA utility with a log case at gamma≈1.
    Returns -inf if c <= 0 (so infeasible candidates automatically lose).
    """
    c = float(c)
    if not np.isfinite(c) or c <= 0.0:
        return float("-inf")

    # avoid log(0) / overflow from tiny c
    c = max(c, eps)

    if np.isclose(gamma, 1.0):
        return float(np.log(c))
    return float(np.exp((1.0 - gamma) * np.log(c)) / (1.0 - gamma))


def node_flow_and_drift(
    grid: Any,
    par: Any,
    prim: Any,
    s: int,
    node: Tuple[int, int],
    tau: float,
    h: float,
    T: float,
    omega: np.ndarray,
    *,
    eps_c: float = 1e-8,
    require_feasible: bool = True,
) -> tuple[float, float, float]:
    """
    Step 4 (Track B): node-local flow payoff and state drifts (kdot, Ldot).

    Returns:
        (flow, kdot, Ldot)

    If require_feasible=True and the node/control implies invalid objects,
    returns (nan, nan, nan) so problems are obvious during debugging.
    """
    i, j = node
    k = float(grid.k[i])
    L = float(grid.L[j])
    tau = float(tau)
    h = float(h)
    T = float(T)

    # omega at node
    omg = float(omega[i, j]) if np.ndim(omega) == 2 else float("nan")

    # Basic sanity
    if not (np.isfinite(k) and np.isfinite(L) and np.isfinite(tau) and np.isfinite(h) and np.isfinite(T) and np.isfinite(omg)):
        return (float("nan"), float("nan"), float("nan")) if require_feasible else (float("-inf"), 0.0, 0.0)

    # Production/pricing
    Y, w, r_k = production_block(k, s, par)

    # Safe rate pinned down by (tau, h) and state (k, L)
    r_f = safe_rate(k=k, L=L, h=h, tau=tau, r_k=r_k, par=par)
    if not np.isfinite(r_f):
        return (float("nan"), float("nan"), float("nan")) if require_feasible else (float("-inf"), 0.0, 0.0)

    # Consumption
    C_W = float(w + T)
    C_K = float(omg * (k + L))

    # Hard positivity (matches what static_feasible_node is trying to enforce)
    if C_W <= eps_c or C_K <= eps_c:
        return (float("nan"), float("nan"), float("nan")) if require_feasible else (float("-inf"), 0.0, 0.0)

    # Flow payoff (utilitarian weights)
    u_W = crra_utility(C_W, par.gamma)
    u_K = crra_utility(C_K, par.gamma)
    flow = float(par.chi * u_W + (1.0 - par.chi) * u_K)

    # Drifts
    kdot = float(Y - C_W - C_K - (par.delta + par.g) * k)

    B = float(L + h)  # gross debt
    Ldot = float((r_f * B) + T - (h * r_k) - tau * ((k - h) * r_k + r_f * B))

    # Final sanity (optional but useful while debugging)
    if require_feasible and not (np.isfinite(flow) and np.isfinite(kdot) and np.isfinite(Ldot)):
        return (float("nan"), float("nan"), float("nan"))

    return flow, kdot, Ldot


# In[13]:


# ============================================================
# COMPLETE REPLACEMENT FOR CELL 13
# Track B — Step 7: Policy improvement + Analytic Transfers + Active Omega
# ============================================================
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, Dict, Iterator, Optional, Tuple

NEG_INF = -1.0e300

# -----------------------------
# 1. Base Utilities & Projections
# -----------------------------
def iter_nodes_where_legacy_b(mask: np.ndarray) -> Iterator[Tuple[int, int]]:
    ii, jj = np.where(mask)
    for i, j in zip(ii, jj): yield (int(i), int(j))

def empty_policy_like_legacy_b(mask: np.ndarray) -> Dict[str, np.ndarray]:
    shape = mask.shape
    return {"tau": np.full(shape, np.nan), "h": np.full(shape, np.nan), "T": np.full(shape, np.nan)}

def mask_policy_legacy_b(u: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in u.items()}
    for key in out: out[key][~mask] = np.nan
    return out

def _I_s(par: Any, s: int) -> float:
    return float(par.I1 if int(s) == 1 else par.I0)

def _wage_vector(grid: Any, par: Any, s: int) -> np.ndarray:
    k = np.asarray(grid.k, dtype=float)
    I = _I_s(par, s)
    return (1.0 - I) * (Phi(I) * np.power(k, I))

def project_policy_state_dependent(u, grid, par, prim, s, mask=None, *, eps_c=1e-8):
    out = {k: np.array(v, copy=True) for k, v in u.items()}
    if mask is None: mask = np.ones(grid.shape, dtype=bool)
    
    out["tau"][mask] = np.clip(out["tau"][mask], prim.tau_min, prim.tau_max)
    
    k_grid = np.asarray(grid.k)[:, None]
    L_grid = np.asarray(grid.L)[None, :]
    h_lo = np.maximum(0.0, -L_grid)
    out["h"][mask] = np.minimum(np.maximum(out["h"], h_lo), k_grid)[mask]
    
    w_k = _wage_vector(grid, par, s)[:, None]
    T_lo = np.maximum(prim.T_min, -w_k + float(eps_c))
    out["T"][mask] = np.minimum(np.maximum(out["T"], T_lo), prim.T_max)[mask]
    return out

def blend_and_project_on_mask(u_old, u_targ, eta, grid, par, prim, s, mask, *, eps_c=1e-8):
    out = {}
    for k in ("tau", "h", "T"):
        out[k] = np.where(mask, (1.0 - eta) * u_old[k] + eta * u_targ[k], u_old[k])
    return project_policy_state_dependent(out, grid, par, prim, s, mask=mask, eps_c=eps_c)

def select_blend_or_snap(u_blend, u_targ, ok_mask):
    return {k: np.where(ok_mask, u_blend[k], u_targ[k]) for k in ("tau", "h", "T")}

def policy_supnorm(u_new, u_old, mask):
    d = 0.0
    for k in ("tau", "h", "T"):
        diff = np.abs(u_new[k] - u_old[k])
        if np.any(mask): d = max(d, float(np.nanmax(diff[mask])))
    return d

def drift(grid, par, prim, s, u, omega):
    k, L = np.asarray(grid.k)[:, None], np.asarray(grid.L)[None, :]
    tau, h, T, omg = u["tau"], u["h"], u["T"], np.asarray(omega)
    I = _I_s(par, s)
    Y = Phi(I) * np.power(k, I)
    w = (1.0 - I) * Y
    r_k = (I * (Y / k)) - par.delta
    W = k + L
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_mc = (k - h) / W
        pi_mc = np.where(W > 0, pi_mc, 0.0)
        
    r_f = r_k - par.gamma * (1.0 - tau) * (par.sigma ** 2) * pi_mc
    
    kdot = Y - (w + T) - (omg * W) - (par.delta + par.g) * k
    Ldot = (r_f * (L + h)) + T - (h * r_k) - tau * (((k - h) * r_k) + (r_f * (L + h)))
    return kdot, Ldot

# -----------------------------
# 2. Analytical Transfer Rules (Achdou-Moll)
# -----------------------------
def _upwind_scalar_deriv(drift_val: float, J_f: float, J_b: float, eps: float) -> float:
    if drift_val > eps: return J_f if np.isfinite(J_f) else np.nan
    if drift_val < -eps: return J_b if np.isfinite(J_b) else np.nan
    return 0.0

def AM_transfer_update_legacy_b(
    node, Jx_f, Jx_b, Jy_f, Jy_b, grid, par, prim, s, tau, h, omega, eps_drift, eps_c
) -> float:
    i, j = node
    k = float(grid.k[i])
    _, kdot_0, Ldot_0 = node_flow_and_drift(grid, par, prim, s, node, tau, h, 0.0, omega, require_feasible=False)
    
    Jk = _upwind_scalar_deriv(kdot_0, Jx_f[i, j], Jx_b[i, j], eps_drift)
    JL = _upwind_scalar_deriv(Ldot_0, Jy_f[i, j], Jy_b[i, j], eps_drift)
    if not np.isfinite(Jk): Jk = 0.0
    if not np.isfinite(JL): JL = 0.0
        
    diff = Jk - JL
    if diff <= 1e-12: # Planner strongly prefers capital to worker consumption
        return float(prim.T_max)
        
    _, w, _ = production_block(k, s, par)
    c_w_optimal = par.chi / diff if np.isclose(par.gamma, 1.0) else (diff / par.chi) ** (-1.0 / par.gamma)
    
    T_min_eff = max(float(prim.T_min), float(eps_c - w))
    return float(np.clip(c_w_optimal - w, T_min_eff, prim.T_max))

def inward_rescue_transfer_legacy_b(
    node, grid, par, prim, s, tau, h, omega, eps_drift, active, T_prefer, eps_c
) -> float:
    i, j = node
    _, kdot_0, Ldot_0 = node_flow_and_drift(grid, par, prim, s, node, tau, h, 0.0, omega, require_feasible=False)
    if not (np.isfinite(kdot_0) and np.isfinite(Ldot_0)): return None

    Nx, Ny = active.shape
    block_e = (kdot_0 > eps_drift) and (i == Nx - 1 or not active[i + 1, j])
    block_w = (kdot_0 < -eps_drift) and (i == 0 or not active[i - 1, j])
    block_n = (Ldot_0 > eps_drift) and (j == Ny - 1 or not active[i, j + 1])
    block_s = (Ldot_0 < -eps_drift) and (j == 0 or not active[i, j - 1])
    
    _, w, _ = production_block(float(grid.k[i]), s, par)
    T_min_req, T_max_req = max(float(prim.T_min), float(eps_c - w)), float(prim.T_max)
    
    if block_e: T_min_req = max(T_min_req, kdot_0 - eps_drift) 
    if block_w: T_max_req = min(T_max_req, kdot_0 + eps_drift) 
    if block_n: T_max_req = min(T_max_req, eps_drift - Ldot_0) 
    if block_s: T_min_req = max(T_min_req, -eps_drift - Ldot_0) 

    if T_min_req > T_max_req + 1e-8: return None
    return float(np.clip(T_prefer, T_min_req, T_max_req))

# -----------------------------
# 3. Robust Policy Gatekeeper
# -----------------------------
def policy_improvement_gatekeep_legacy_b(grid, par, prim, s, J, omega, M, *, eps_drift=1e-12, eps_c=1e-8):
    active = np.asarray(M, dtype=bool)
    Jx_f, Jx_b, Jy_f, Jy_b = masked_upwind_derivatives(np.asarray(J, dtype=float), active, grid)

    u_target = empty_policy_like(active)
    M_target = np.zeros_like(active, dtype=bool)

    tau_grid = [float(x) for x in prim.tau_grid]
    h_grid   = [float(x) for x in prim.h_grid]

    for node in iter_nodes_where(active):
        i, j = node
        best_H, best_u = NEG_INF, None

        for tau in tau_grid:
            for h in h_grid:
                # 1. Analytical Proposal
                T_am = AM_transfer_update(node, Jx_f, Jx_b, Jy_f, Jy_b, grid, par, prim, s, tau, h, omega, eps_drift, eps_c)
                if not static_feasible_node(grid, par, prim, s, node, tau, h, T_am, omega, eps_c=eps_c): continue

                # 2. Gate & Rescue
                flow, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T_am, omega)
                if not inward_one_cell_node(active, node, kdot, Ldot, eps_drift):
                    T_resc = inward_rescue_transfer(node, grid, par, prim, s, tau, h, omega, eps_drift, active, T_am, eps_c)
                    if T_resc is None: continue
                    if not static_feasible_node(grid, par, prim, s, node, tau, h, T_resc, omega, eps_c=eps_c): continue
                    flow, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T_resc, omega)
                    if not inward_one_cell_node(active, node, kdot, Ldot, eps_drift): continue
                    T_use = T_resc
                else:
                    T_use = T_am

                # 3. Calculate Hamiltonian natively
                Jk = _upwind_scalar_deriv(kdot, Jx_f[i, j], Jx_b[i, j], eps_drift)
                JL = _upwind_scalar_deriv(Ldot, Jy_f[i, j], Jy_b[i, j], eps_drift)
                if not (np.isfinite(Jk) and np.isfinite(JL)): continue

                H = float(flow + kdot * Jk + Ldot * JL)
                if H > best_H:
                    best_H, best_u = H, (tau, h, float(T_use))

        if best_u is None:
            M_target[node] = False
            continue

        M_target[node] = True
        u_target["tau"][node], u_target["h"][node], u_target["T"][node] = best_u

    return mask_policy(u_target, M_target), M_target

def improve_with_prune_closure_legacy_b(grid, par, prim, s, J, omega, M, *, eps_drift=1e-12, eps_c=1e-8, max_passes=10):
    M_work = np.asarray(M, dtype=bool).copy()
    u_targ = empty_policy_like(M_work)

    for _ in range(int(max_passes)):
        u_cand, M_cand = policy_improvement_gatekeep(grid, par, prim, s, J, omega, M_work, eps_drift=eps_drift, eps_c=eps_c)
        M_new = M_work & M_cand
        u_targ = u_cand
        if np.array_equal(M_new, M_work): return mask_policy(u_targ, M_work), M_work
        M_work = M_new

    return mask_policy(u_targ, M_work), M_work

# -----------------------------
# 4. Active Omega PDE Solver
# -----------------------------
def update_private_omega_legacy_b(grid, par, prim, s, omega_old, u_s, Momega_s, lam=0.0, omega1_new=None):
    """Solves the capital owners' linear PDE for Psi_s on the active mask Momega_s."""
    active = np.asarray(Momega_s, dtype=bool)
    if not np.any(active): return omega_old.copy()
        
    def node_eval(node):
        i, j = node
        tau, h, T = float(u_s["tau"][node]), float(u_s["h"][node]), float(u_s["T"][node])
        _, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega_old, require_feasible=False)
        
        flow = omega_old[i, j] ** (1.0 - par.gamma)
        if s == 0 and omega1_new is not None: flow += lam * (omega1_new[i, j] ** (-par.gamma))
        return NodeDriftFlow(kdot=float(kdot), Ldot=float(Ldot), flow=float(flow))

    A, f, act = build_masked_system_2_3(grid, active, node_eval, eps_drift=1e-12, check_inward=False)
    
    n_act = act.n_active
    D_shift = np.zeros(n_act)
    for p_act in range(n_act):
        node = unflatten(act.idx_full[p_act], grid)
        i, j = node
        tau, h, k, L = float(u_s["tau"][node]), float(u_s["h"][node]), float(grid.k[i]), float(grid.L[j])
        _, _, r_k = production_block(k, s, par)
        r_f = safe_rate(k, L, h, tau, r_k, par)
        pi_mc = market_clearing_risky_share(k, L, h)
        R_s = (1.0 - tau) * r_f + 0.5 * par.gamma * ((1.0 - tau)**2) * (par.sigma**2) * (pi_mc**2)
        D_shift[p_act] = (1.0 - par.gamma) * (R_s - omega_old[i, j])

    I = sp.eye(n_act, format="csr", dtype=float)
    D_mat = sp.diags(D_shift, format="csr")
    LHS = (par.rho + lam) * I - D_mat - A
    
    Psi_act = spla.spsolve(LHS.tocsc(), f)
    Psi_act = np.maximum(Psi_act, 1e-12) 
    
    Psi_full = embed_active_to_full(Psi_act, act, grid, anchor=np.nan)
    omega_new = omega_old.copy()
    valid = active & np.isfinite(Psi_full)
    omega_new[valid] = Psi_full[valid] ** (-1.0 / par.gamma)
    return omega_new


# In[14]:


# ============================================================
# CONNECTING CODE — Canonical runner (Frozen ω milestone)
# Keep ONE entrypoint to avoid duplicated logic.
# ============================================================

import numpy as np

def _require(names):
    missing = [n for n in names if n not in globals()]
    if missing:
        raise NameError(
            "Missing definitions in this kernel:\n  - "
            + "\n  - ".join(missing)
            + "\n\nRun (i) solver Sections 1–7, and (ii) your economics block notebook."
        )

# Solver core needed
_require([
    "Grid", "Prim",
    "empty_policy_like", "mask_policy",
    "improve_with_prune_closure",
    "outer_loop_solver",
])

# Econ callbacks needed (must be defined by your economics notebook)
_require([
    "static_feasible_node",
    "node_flow_and_drift",
])

# Optional econ primitive feasible set (ω-free); if not present we use a safe fallback below
_HAS_PRIM_FEAS_BASIC = ("primitive_feasible_set_basic" in globals())


def primitive_feasible_set_fallback_legacy_b(grid, par, prim):
    """
    Conservative ω-free feasible set if you haven't defined primitive_feasible_set_basic.
    Enforces:
      k > 0,  k+L > 0,  L >= -k
    Outer solver will also intersect with grid.interior_mask.
    """
    K, LL = np.meshgrid(grid.k, grid.L, indexing="ij")
    return (K > 0.0) & (K + LL > 0.0) & (LL >= -K)


def update_private_omega_frozen_legacy_b(grid, par, prim, s, omega_old, u_s, Momega_s):
    """Frozen ω update (milestone runner)."""
    return np.asarray(omega_old, dtype=float).copy()


def make_owner_domains_legacy_b(base_active: np.ndarray):
    """Default owner domain: base_active."""
    Momega1 = base_active.copy()
    Momega0 = base_active.copy()
    if not np.any(Momega1) or not np.any(Momega0):
        raise ValueError("Owner domain empty; check grid bounds / primitive feasible set.")
    return Momega1, Momega0


def prim_coarse(prim: Prim, n_tau: int = 7, n_h: int = 11) -> Prim:
    """Coarse candidate grids for fast safe initialization."""
    tau = np.asarray(prim.tau_grid, float)
    h   = np.asarray(prim.h_grid, float)

    def _sub(x, n):
        if x.size <= n:
            return x
        idx = np.linspace(0, x.size - 1, n).round().astype(int)
        return x[idx]

    return Prim(
        tau_grid=_sub(tau, n_tau),
        h_grid=_sub(h, n_h),
        T_min=prim.T_min, T_max=prim.T_max,
        tau_min=prim.tau_min, tau_max=prim.tau_max,
        h_min=prim.h_min, h_max=prim.h_max,
    )


def initialize_policy_safe_legacy_b(grid, par, prim: Prim, s: int, omega: np.ndarray, M_init: np.ndarray,
                           T_grid: np.ndarray, eps_drift: float = 1e-12,
                           coarse_init: bool = True, verbose: bool = True):
    """
    Safe initialization via improve_with_prune_closure on J=0.
    Returns (u_init, M_stable).
    """
    prim_use = prim_coarse(prim) if coarse_init else prim
    J0 = np.zeros(grid.shape)

    if verbose:
        print(f"[init] s={s} |M_init|={int(M_init.sum())} coarse={coarse_init}")

    u_targ, M_stable = improve_with_prune_closure(
        grid, par, prim_use,
        s=s, J=J0, omega=omega, M=M_init,
        static_feasible_node=static_feasible_node,
        node_flow_and_drift=node_flow_and_drift,
        eps_drift=eps_drift,
        transfer_rule=None, rescue_transfer=None,
        T_grid=T_grid,
        max_passes=25,
        verbose=False,
    )
    u_targ = mask_policy(u_targ, M_stable)

    if verbose:
        print(f"[init] s={s} |M_stable|={int(M_stable.sum())}")

    return u_targ, M_stable


def run_planner_frozen_omega_legacy_b(grid, par, prim: Prim, *,
                             lam: float,
                             omega_level: float = 0.05,
                             T_grid: np.ndarray = None,
                             primitive_feasible_set_fn=None,
                             eps_drift: float = 1e-12,
                             zeta_omega: float = 0.5,
                             N_peel: int = 3,
                             max_outer: int = 10,
                             tol_outer: float = 1e-8,
                             eta_policy: float = 0.8,
                             m_inner_max: int = 25,
                             tol_policy: float = 1e-7,
                             coarse_init: bool = True,
                             verbose: bool = True):
    """
    Canonical milestone runner:
      - uses your real static_feasible_node + node_flow_and_drift
      - freezes ω (so private block not required yet)
      - runs full outer_loop_solver plumbing
    """
    if T_grid is None:
        T_grid = np.array([prim.T_min, 0.0, prim.T_max], dtype=float)
    else:
        T_grid = np.asarray(T_grid, float).reshape(-1)

    if primitive_feasible_set_fn is None:
        if _HAS_PRIM_FEAS_BASIC:
            primitive_feasible_set_fn = globals()["primitive_feasible_set_basic"]
        else:
            primitive_feasible_set_fn = primitive_feasible_set_fallback

    S = np.asarray(primitive_feasible_set_fn(grid, par, prim), dtype=bool)
    base_active = S & grid.interior_mask
    if not np.any(base_active):
        raise ValueError("base_active empty: check grid bounds + primitive feasibility.")

    Momega1, Momega0 = make_owner_domains(base_active)

    omega1_init = np.full(grid.shape, float(omega_level))
    omega0_init = np.full(grid.shape, float(omega_level))

    # safe init masks
    M1_init = base_active.copy()
    M0_init = base_active.copy()
    M0_init &= M1_init

    # safe init policies
    u1_init, M1_stable = initialize_policy_safe(
        grid, par, prim, s=1, omega=omega1_init, M_init=M1_init,
        T_grid=T_grid, eps_drift=eps_drift,
        coarse_init=coarse_init, verbose=verbose
    )
    u0_init, M0_stable = initialize_policy_safe(
        grid, par, prim, s=0, omega=omega0_init, M_init=(M0_init & M1_stable),
        T_grid=T_grid, eps_drift=eps_drift,
        coarse_init=coarse_init, verbose=verbose
    )
    M0_stable &= M1_stable

    J1_init = np.zeros(grid.shape)
    J0_init = np.zeros(grid.shape)

    if verbose:
        print("\n[frozen ω] starting outer loop")
        print("  |base_active| =", int(base_active.sum()))
        print("  |M1_stable|   =", int(M1_stable.sum()))
        print("  |M0_stable|   =", int(M0_stable.sum()))

    sol = outer_loop_solver(
        grid, par, prim,
        lam=lam,
        omega1_init=omega1_init, omega0_init=omega0_init,
        J1_init=J1_init, J0_init=J0_init,
        u1_init=u1_init, u0_init=u0_init,
        Momega1=Momega1, Momega0=Momega0,
        primitive_feasible_set=primitive_feasible_set_fn,
        update_private_omega=update_private_omega_frozen,
        static_feasible_node=static_feasible_node,
        node_flow_and_drift=node_flow_and_drift,
        zeta_omega=zeta_omega,
        N_peel=N_peel,
        eps_drift=eps_drift,
        max_outer=max_outer,
        tol_outer=tol_outer,
        eta_policy=eta_policy,
        m_inner_max=m_inner_max,
        tol_policy=tol_policy,
        T_grid=T_grid,
        do_full_peel=False,
        verbose=verbose,
    )

    if verbose:
        print("\n[frozen ω] done")
        print("  outer iters =", len(sol["history"]))
        print("  final |M1|  =", int(sol["M1"].sum()))
        print("  final |M0|  =", int(sol["M0"].sum()))

    return sol


# In[15]:


# ============================================================
# COMPLETE REPLACEMENT FOR CELL 15
# Fixes the missing 'drift_node' and aligns all solver loops
# ============================================================
import numpy as np

# ---------------------------------------------------------
# 1. Missing Adapters and Fallbacks
# ---------------------------------------------------------
def drift_node(grid, par, prim, s, node, tau, h, T, omega):
    """Adapter to give viability peeling the (kdot, Ldot) tuple it expects."""
    _, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega, require_feasible=False)
    return float(kdot), float(Ldot)

def _core_max_norm(delta: np.ndarray, core: np.ndarray, label: str) -> float:
    if not np.any(core): return 0.0
    return float(np.nanmax(np.abs(delta[core])))

def primitive_feasible_set_fallback_legacy_c(grid, par, prim):
    """Fallback no-default domain: k > 0, owner wealth >= 0, L >= -k."""
    K, LL = np.meshgrid(grid.k, grid.L, indexing="ij")
    return (K > 0.0) & (K + LL > 0.0) & (LL >= -K)

def make_owner_domains_legacy_c(base_active):
    return base_active.copy(), base_active.copy()

def update_private_omega_frozen_legacy_c(grid, par, prim, s, omega_old, u_s, Momega_s):
    """Returns omega unchanged (for the frozen milestone)."""
    return np.asarray(omega_old, dtype=float).copy()


# ---------------------------------------------------------
# 2. Howard Inner Loop
# ---------------------------------------------------------
def howard_inner_loop_legacy_c(
    grid, par, prim, *, lam,
    omega1, omega0, J1_init, J0_init, u1_init, u0_init, M1_init, M0_init,
    eta_policy=0.8, eps_drift=1e-12, m_inner_max=40, tol_policy=1e-7, verbose=False
):
    M1 = np.asarray(M1_init, dtype=bool).copy()
    M0 = np.asarray(M0_init, dtype=bool).copy()
    M0 &= M1  

    J1, J0 = J1_init.copy(), J0_init.copy()
    u1 = {k: np.asarray(v, dtype=float).copy() for k, v in u1_init.items()}
    u0 = {k: np.asarray(v, dtype=float).copy() for k, v in u0_init.items()}

    for m in range(m_inner_max):
        # (A) Evaluate Regime 1
        def ne1(node):
            tau, h, T = float(u1["tau"][node]), float(u1["h"][node]), float(u1["T"][node])
            flow, kd, Ld = node_flow_and_drift(grid, par, prim, 1, node, tau, h, T, omega1)
            return NodeDriftFlow(kdot=float(kd), Ldot=float(Ld), flow=float(flow))
            
        A1, f1, act1 = build_masked_system_2_3(grid, M1, ne1, eps_drift=eps_drift, check_inward=False)
        J1_new = embed_active_to_full(solve_hjb_on_active(A1, f1, rho=par.rho), act1, grid, anchor=np.nan)

        # (A) Evaluate Regime 0
        def ne0(node):
            tau, h, T = float(u0["tau"][node]), float(u0["h"][node]), float(u0["T"][node])
            flow, kd, Ld = node_flow_and_drift(grid, par, prim, 0, node, tau, h, T, omega0)
            return NodeDriftFlow(kdot=float(kd), Ldot=float(Ld), flow=float(flow))
            
        A0, f0, act0 = build_masked_system_2_3(grid, M0, ne0, eps_drift=eps_drift, check_inward=False)
        J1_on_M0 = restrict_full_to_active(J1_new, act0, grid)
        rhs0 = f0 + lam * J1_on_M0
        J0_new = embed_active_to_full(solve_hjb_on_active(A0, rhs0, rho=par.rho, lam=lam), act0, grid, anchor=np.nan)

        # (B) Improve with Prune Closure 
        u1_targ, M1_stable = improve_with_prune_closure(grid, par, prim, 1, J1_new, omega1, M1, eps_drift=eps_drift, max_passes=10)
        u0_targ, M0_stable = improve_with_prune_closure(grid, par, prim, 0, J0_new, omega0, M0, eps_drift=eps_drift, max_passes=10)
        M0_stable &= M1_stable

        u1_targ = mask_policy(u1_targ, M1_stable)
        u0_targ = mask_policy(u0_targ, M0_stable)

        # (C) Damp & Project
        u1_blend = blend_and_project_on_mask(u1, u1_targ, eta_policy, grid, par, prim, 1, M1_stable)
        u0_blend = blend_and_project_on_mask(u0, u0_targ, eta_policy, grid, par, prim, 0, M0_stable)

        # Gate
        kdot1, Ldot1 = drift(grid, par, prim, 1, u1_blend, omega1)
        ok1 = inward_one_cell(M1_stable, kdot1, Ldot1, eps=eps_drift)

        kdot0, Ldot0 = drift(grid, par, prim, 0, u0_blend, omega0)
        ok0 = inward_one_cell(M0_stable, kdot0, Ldot0, eps=eps_drift)

        u1_next = mask_policy(select_blend_or_snap(u1_blend, u1_targ, ok1), M1_stable)
        u0_next = mask_policy(select_blend_or_snap(u0_blend, u0_targ, ok0), M0_stable)

        # (D) Stopping
        pol_diff = policy_supnorm(u1_next, u1, M1_stable) + policy_supnorm(u0_next, u0, M0_stable)
        mask_same = np.array_equal(M1_stable, M1) and np.array_equal(M0_stable, M0)

        if verbose: print(f"  [Howard] iter {m+1} pol_diff={pol_diff:.3e}")

        if pol_diff < tol_policy and mask_same:
            return J1_new, J0_new, u1_next, u0_next, M1_stable, M0_stable

        J1, J0, u1, u0, M1, M0 = J1_new, J0_new, u1_next, u0_next, M1_stable, (M0_stable & M1_stable)

    return J1, J0, u1, u0, M1, M0


# ---------------------------------------------------------
# 3. Outer Loop Solver
# ---------------------------------------------------------
def outer_loop_solver_legacy_c(
    grid, par, prim, *, lam,
    omega1_init, omega0_init, J1_init, J0_init, u1_init, u0_init, Momega1, Momega0,
    primitive_feasible_set_fn, update_private_omega,
    zeta_omega=0.2, N_peel=5, eps_drift=1e-12, max_outer=200, tol_outer=1e-6,
    stable_window=3, min_mask_size=10, eta_policy=0.8, m_inner_max=40, tol_policy=1e-7, verbose=False,
):
    S = np.asarray(primitive_feasible_set_fn(grid, par, prim), dtype=bool)
    B = np.asarray(grid.interior_mask, dtype=bool)
    base_active = S & B

    V1, V0 = base_active.copy(), base_active.copy()
    V0 &= V1
    M1, M0 = V1.copy(), V0.copy()

    omega1, omega0 = omega1_init.copy(), omega0_init.copy()
    J1, J0 = J1_init.copy(), J0_init.copy()
    u1 = {k: v.copy() for k,v in u1_init.items()}
    u0 = {k: v.copy() for k,v in u0_init.items()}

    history = []
    M1_prev, M0_prev = None, None
    stable_count = 0
    Momega1_eff_prev, Momega0_eff_prev = None, None

    for n in range(max_outer):
        Momega1_eff, Momega0_eff = Momega1 & M1, Momega0 & M0

        omega1_new = update_private_omega(grid, par, prim, 1, omega1, u1, Momega1_eff)
        omega0_new = update_private_omega(grid, par, prim, 0, omega0, u0, Momega0_eff)

        omega1_half = (1.0 - zeta_omega) * omega1 + zeta_omega * omega1_new
        omega0_half = (1.0 - zeta_omega) * omega0 + zeta_omega * omega0_new

        omega1_ext = quarantine_fill_nearest(omega1_half, Momega1_eff)
        omega0_ext = quarantine_fill_nearest(omega0_half, Momega0_eff)

        V1 = viability_peel_warm(V1, base_active, grid, par, prim, omega1_ext, 1, static_feasible_node=static_feasible_node, drift_node=drift_node, N_peel=N_peel, eps_drift=eps_drift)
        V0 = viability_peel_warm(V0, base_active, grid, par, prim, omega0_ext, 0, static_feasible_node=static_feasible_node, drift_node=drift_node, N_peel=N_peel, eps_drift=eps_drift)
        V0 &= V1

        M1, M0 = V1.copy(), V0.copy()

        J1_new, J0_new, u1_new, u0_new, M1_new, M0_new = howard_inner_loop(
            grid, par, prim, lam=lam, omega1=omega1_ext, omega0=omega0_ext,
            J1_init=J1, J0_init=J0, u1_init=u1, u0_init=u0, M1_init=M1, M0_init=M0,
            eta_policy=eta_policy, eps_drift=eps_drift, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=False
        )

        core1, core0 = M1 & M1_new, M0 & M0_new
        core_om1 = Momega1_eff if (Momega1_eff_prev is None) else (Momega1_eff & Momega1_eff_prev)
        core_om0 = Momega0_eff if (Momega0_eff_prev is None) else (Momega0_eff & Momega0_eff_prev)

        d_omega = _core_max_norm(omega1_ext - omega1, core_om1, label="d_omega1") + _core_max_norm(omega0_ext - omega0, core_om0, label="d_omega0")
        d_J = _core_max_norm(J1_new - J1, core1, label="d_J1") + _core_max_norm(J0_new - J0, core0, label="d_J0")
        d_u = policy_supnorm(u1_new, u1, core1) + policy_supnorm(u0_new, u0, core0)

        history.append({"outer_iter": n, "d_omega": float(d_omega), "d_J": float(d_J), "d_u": float(d_u), "size_M1": int(M1_new.sum()), "size_M0": int(M0_new.sum())})

        if verbose: print(f"[Outer {n+1}/{max_outer}] d_omega={d_omega:.3e} d_J={d_J:.3e} d_u={d_u:.3e} |M1|={int(M1_new.sum())}")

        if (M1_prev is not None) and np.array_equal(M1_new, M1_prev) and np.array_equal(M0_new, M0_prev):
            stable_count += 1
        else:
            stable_count = 0
        
        M1_prev, M0_prev = M1_new.copy(), M0_new.copy()

        omega1, omega0 = omega1_ext, omega0_ext
        J1, J0, u1, u0 = J1_new, J0_new, u1_new, u0_new
        Momega1_eff_prev, Momega0_eff_prev = Momega1_eff.copy(), Momega0_eff.copy()

        V1 &= M1_new
        V0 &= (M0_new & V1)
        M1, M0 = V1.copy(), V0.copy()

        if (d_omega < tol_outer) and (d_J < tol_outer) and (d_u < tol_outer) and (stable_count >= stable_window):
            if verbose: print("[Outer] Converged.")
            break

    return {"omega1": omega1, "omega0": omega0, "J1": J1, "J0": J0, "u1": u1, "u0": u0, "M1": M1.copy(), "M0": M0.copy(), "history": history}


# ---------------------------------------------------------
# 4. Safe Initializer
# ---------------------------------------------------------
def initialize_policy_safe_legacy_c(grid, par, prim, s, omega, M_init, eps_drift=1e-12, coarse_init=True):
    import copy
    prim_use = copy.copy(prim)
    if coarse_init:
        prim_use.tau_grid = np.linspace(prim.tau_min, prim.tau_max, 5)
        prim_use.h_grid = np.linspace(prim.h_min, prim.h_max, 5)
    
    J0 = np.zeros(grid.shape)
    u_targ, M_stable = improve_with_prune_closure(grid, par, prim_use, s, J0, omega, M_init, eps_drift=eps_drift, max_passes=25)
    return mask_policy(u_targ, M_stable), M_stable


# ---------------------------------------------------------
# 5. Milestone Runners (Frozen vs Active Omega)
# ---------------------------------------------------------
def run_planner_frozen_omega_legacy_c(grid, par, prim, *, lam, omega_level=0.05, eps_drift=1e-12, zeta_omega=0.5, N_peel=3, max_outer=10, tol_outer=1e-8, eta_policy=0.8, m_inner_max=25, tol_policy=1e-7, coarse_init=True, verbose=True):
    S = np.asarray(primitive_feasible_set_fallback(grid, par, prim), dtype=bool)
    base_active = S & grid.interior_mask
    Momega1, Momega0 = make_owner_domains(base_active)
    omega1_init, omega0_init = np.full(grid.shape, float(omega_level)), np.full(grid.shape, float(omega_level))
    
    u1_init, M1_stable = initialize_policy_safe(grid, par, prim, 1, omega1_init, base_active.copy(), eps_drift=eps_drift, coarse_init=coarse_init)
    u0_init, M0_stable = initialize_policy_safe(grid, par, prim, 0, omega0_init, base_active.copy() & M1_stable, eps_drift=eps_drift, coarse_init=coarse_init)
    M0_stable &= M1_stable

    if verbose: print(f"\n[frozen ω] Starting outer loop |M1|={int(M1_stable.sum())} |M0|={int(M0_stable.sum())}")

    return outer_loop_solver(
        grid, par, prim, lam=lam, omega1_init=omega1_init, omega0_init=omega0_init,
        J1_init=np.zeros(grid.shape), J0_init=np.zeros(grid.shape), u1_init=u1_init, u0_init=u0_init,
        Momega1=Momega1, Momega0=Momega0, primitive_feasible_set_fn=primitive_feasible_set_fallback,
        update_private_omega=update_private_omega_frozen, 
        zeta_omega=zeta_omega, N_peel=N_peel, 
        max_outer=max_outer, tol_outer=tol_outer,
        eta_policy=eta_policy, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=verbose
    )

def run_planner_active_omega_legacy_c(grid, par, prim, *, lam, omega_level=0.05, eps_drift=1e-12, zeta_omega=0.5, N_peel=3, max_outer=10, tol_outer=1e-8, eta_policy=0.8, m_inner_max=25, tol_policy=1e-7, coarse_init=True, verbose=True):
    S = np.asarray(primitive_feasible_set_fallback(grid, par, prim), dtype=bool)
    base_active = S & grid.interior_mask
    Momega1, Momega0 = make_owner_domains(base_active)
    omega1_init, omega0_init = np.full(grid.shape, float(omega_level)), np.full(grid.shape, float(omega_level))
    
    u1_init, M1_stable = initialize_policy_safe(grid, par, prim, 1, omega1_init, base_active.copy(), eps_drift=eps_drift, coarse_init=coarse_init)
    u0_init, M0_stable = initialize_policy_safe(grid, par, prim, 0, omega0_init, base_active.copy() & M1_stable, eps_drift=eps_drift, coarse_init=coarse_init)
    M0_stable &= M1_stable

    if verbose: print(f"\n[active ω] Starting outer loop |M1|={int(M1_stable.sum())} |M0|={int(M0_stable.sum())}")

    last_omega1_new = [omega1_init]
    def wrapped_update_omega(g, p, pr, s, omg_old, u, M_omg):
        if s == 1:
            omg1 = update_private_omega(g, p, pr, 1, omg_old, u, M_omg, lam=0.0, omega1_new=None)
            last_omega1_new[0] = omg1
            return omg1
        else:
            return update_private_omega(g, p, pr, 0, omg_old, u, M_omg, lam=lam, omega1_new=last_omega1_new[0])

    return outer_loop_solver(
        grid, par, prim, lam=lam, omega1_init=omega1_init, omega0_init=omega0_init,
        J1_init=np.zeros(grid.shape), J0_init=np.zeros(grid.shape), u1_init=u1_init, u0_init=u0_init,
        Momega1=Momega1, Momega0=Momega0, primitive_feasible_set_fn=primitive_feasible_set_fallback,
        update_private_omega=wrapped_update_omega, 
        zeta_omega=zeta_omega, N_peel=N_peel, 
        max_outer=max_outer, tol_outer=tol_outer,
        eta_policy=eta_policy, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=verbose
    )


# In[16]:


# ============================================================
# MASTER INTEGRATION CELL (Fixes Mask Collapse & Adapters)
# Overrides Step 7 Gatekeepers, Omega Solver, and Outer Loops
# ============================================================
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, Dict, Iterator, Optional, Tuple

NEG_INF = -1.0e300

# -----------------------------
# 1. Base Utilities & Adapters
# -----------------------------
def iter_nodes_where(mask: np.ndarray) -> Iterator[Tuple[int, int]]:
    ii, jj = np.where(mask)
    for i, j in zip(ii, jj): yield (int(i), int(j))

def empty_policy_like(mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {"tau": np.full(mask.shape, np.nan), "h": np.full(mask.shape, np.nan), "T": np.full(mask.shape, np.nan)}

def mask_policy(u: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in u.items()}
    for key in out: out[key][~mask] = np.nan
    return out

def _I_s(par: Any, s: int) -> float:
    return float(par.I1 if int(s) == 1 else par.I0)

def drift_node(grid, par, prim, s, node, tau, h, T, omega):
    """Adapter to give viability peeling the (kdot, Ldot) tuple it expects."""
    _, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega, require_feasible=False)
    return float(kdot), float(Ldot)

def _core_max_norm(delta: np.ndarray, core: np.ndarray, label: str) -> float:
    if not np.any(core): return 0.0
    return float(np.nanmax(np.abs(delta[core])))

def primitive_feasible_set_fallback(grid, par, prim):
    K, LL = np.meshgrid(grid.k, grid.L, indexing="ij")
    return (K > 0.0) & (K + LL > 0.0) & (LL >= -K)

def make_owner_domains(base_active):
    return base_active.copy(), base_active.copy()

def update_private_omega_frozen(grid, par, prim, s, omega_old, u_s, Momega_s):
    return np.asarray(omega_old, dtype=float).copy()

# -----------------------------
# 2. Analytical Transfer Rules
# -----------------------------
def _upwind_scalar_deriv(drift_val: float, J_f: float, J_b: float, eps: float) -> float:
    if drift_val > eps: return J_f if np.isfinite(J_f) else np.nan
    if drift_val < -eps: return J_b if np.isfinite(J_b) else np.nan
    return 0.0

def AM_transfer_update(node, Jx_f, Jx_b, Jy_f, Jy_b, grid, par, prim, s, tau, h, omega, eps_drift, eps_c) -> float:
    i, j = node
    k = float(grid.k[i])
    _, kdot_0, Ldot_0 = node_flow_and_drift(grid, par, prim, s, node, tau, h, 0.0, omega, require_feasible=False)
    
    Jk = _upwind_scalar_deriv(kdot_0, Jx_f[i, j], Jx_b[i, j], eps_drift)
    JL = _upwind_scalar_deriv(Ldot_0, Jy_f[i, j], Jy_b[i, j], eps_drift)
    if not np.isfinite(Jk): Jk = 0.0
    if not np.isfinite(JL): JL = 0.0
        
    diff = Jk - JL
    if diff <= 1e-12: return float(prim.T_max)
        
    _, w, _ = production_block(k, s, par)
    c_w_optimal = par.chi / diff if np.isclose(par.gamma, 1.0) else (diff / par.chi) ** (-1.0 / par.gamma)
    return float(np.clip(c_w_optimal - w, max(float(prim.T_min), float(eps_c - w)), prim.T_max))

def inward_rescue_transfer(node, grid, par, prim, s, tau, h, omega, eps_drift, active, T_prefer, eps_c) -> float:
    """FIXED: Accurately calculates bounding walls and clamps transfers to restore inward drift."""
    i, j = node
    _, kdot_0, Ldot_0 = node_flow_and_drift(grid, par, prim, s, node, tau, h, 0.0, omega, require_feasible=False)
    if not (np.isfinite(kdot_0) and np.isfinite(Ldot_0)): return None

    Nx, Ny = active.shape
    # FIX: Blocked directions must not depend on kdot_0/Ldot_0, because T changes drift!
    wall_e = (i == Nx - 1 or not active[i + 1, j])
    wall_w = (i == 0      or not active[i - 1, j])
    wall_n = (j == Ny - 1 or not active[i, j + 1])
    wall_s = (j == 0      or not active[i, j - 1])
    
    try:
        _, w, _ = production_block(float(grid.k[i]), s, par)
    except Exception:
        w = 0.0
        
    T_min_req = max(float(prim.T_min), float(eps_c - w))
    T_max_req = float(prim.T_max)
    
    # Missing walls unconditionally restrict the transfer T
    if wall_e: T_min_req = max(T_min_req, kdot_0 - eps_drift) 
    if wall_w: T_max_req = min(T_max_req, kdot_0 + eps_drift) 
    if wall_n: T_max_req = min(T_max_req, eps_drift - Ldot_0) 
    if wall_s: T_min_req = max(T_min_req, -eps_drift - Ldot_0) 

    if T_min_req > T_max_req + 1e-8: return None
    return float(np.clip(T_prefer, T_min_req, T_max_req))

# -----------------------------
# 3. Robust Policy Gatekeeper
# -----------------------------
def policy_improvement_gatekeep(grid, par, prim, s, J, omega, M, *, eps_drift=1e-12, eps_c=1e-8, T_grid=None):
    active = np.asarray(M, dtype=bool)
    Jx_f, Jx_b, Jy_f, Jy_b = masked_upwind_derivatives(np.asarray(J, dtype=float), active, grid)

    u_target = empty_policy_like(active)
    M_target = np.zeros_like(active, dtype=bool)

    tau_grid = [float(x) for x in prim.tau_grid]
    T_backup = [float(prim.T_min), 0.0, float(prim.T_max)] if T_grid is None else [float(x) for x in T_grid]

    for node in iter_nodes_where(active):
        i, j = node
        best_H, best_u = NEG_INF, None
        
        k_val, L_val = float(grid.k[i]), float(grid.L[j])
        
        # Dynamically inject local hard solvency boundaries into the h search!
        # This prevents mask collapse at edge cases.
        h_lo, h_hi = max(0.0, -L_val), k_val
        h_cands = [float(x) for x in prim.h_grid if h_lo <= float(x) <= h_hi]
        if h_lo <= h_hi: h_cands.extend([h_lo, h_hi])
        h_cands = list(set(h_cands))

        for tau in tau_grid:
            for h in h_cands:
                T_cands = list(T_backup)
                T_am = AM_transfer_update(node, Jx_f, Jx_b, Jy_f, Jy_b, grid, par, prim, s, tau, h, omega, eps_drift, eps_c)
                T_cands.append(T_am)
                
                for T in set(T_cands):
                    if not static_feasible_node(grid, par, prim, s, node, tau, h, T, omega, eps_c=eps_c): continue

                    flow, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega)
                    if not inward_one_cell_node(active, node, kdot, Ldot, eps_drift):
                        T_resc = inward_rescue_transfer(node, grid, par, prim, s, tau, h, omega, eps_drift, active, T, eps_c)
                        if T_resc is None: continue
                        if not static_feasible_node(grid, par, prim, s, node, tau, h, T_resc, omega, eps_c=eps_c): continue
                        flow2, kdot2, Ldot2 = node_flow_and_drift(grid, par, prim, s, node, tau, h, T_resc, omega)
                        if not inward_one_cell_node(active, node, kdot2, Ldot2, eps_drift): continue
                        flow, kdot, Ldot, T_use = flow2, kdot2, Ldot2, T_resc
                    else:
                        T_use = T

                    Jk = _upwind_scalar_deriv(kdot, Jx_f[i, j], Jx_b[i, j], eps_drift)
                    JL = _upwind_scalar_deriv(Ldot, Jy_f[i, j], Jy_b[i, j], eps_drift)
                    if not (np.isfinite(Jk) and np.isfinite(JL)): continue

                    H_val = float(flow + kdot * Jk + Ldot * JL)
                    if H_val > best_H:
                        best_H, best_u = H_val, (tau, h, float(T_use))

        if best_u is None:
            M_target[node] = False
            continue

        M_target[node] = True
        u_target["tau"][node], u_target["h"][node], u_target["T"][node] = best_u

    return mask_policy(u_target, M_target), M_target

def improve_with_prune_closure(grid, par, prim, s, J, omega, M, *, eps_drift=1e-12, eps_c=1e-8, max_passes=10, T_grid=None):
    M_work = np.asarray(M, dtype=bool).copy()
    u_targ = empty_policy_like(M_work)

    for _ in range(int(max_passes)):
        u_cand, M_cand = policy_improvement_gatekeep(grid, par, prim, s, J, omega, M_work, eps_drift=eps_drift, eps_c=eps_c, T_grid=T_grid)
        M_new = M_work & M_cand
        u_targ = u_cand
        if np.array_equal(M_new, M_work): return mask_policy(u_targ, M_work), M_work
        M_work = M_new

    return mask_policy(u_targ, M_work), M_work

# -----------------------------
# 4. Active Omega PDE Solver
# -----------------------------
def update_private_omega(grid, par, prim, s, omega_old, u_s, Momega_s, lam=0.0, omega1_new=None):
    active = np.asarray(Momega_s, dtype=bool)
    if not np.any(active): return omega_old.copy()
        
    def node_eval(node):
        i, j = node
        tau, h, T = float(u_s["tau"][node]), float(u_s["h"][node]), float(u_s["T"][node])
        _, kdot, Ldot = node_flow_and_drift(grid, par, prim, s, node, tau, h, T, omega_old, require_feasible=False)
        
        flow = omega_old[i, j] ** (1.0 - par.gamma)
        if s == 0 and omega1_new is not None: flow += lam * (omega1_new[i, j] ** (-par.gamma))
        return NodeDriftFlow(kdot=float(kdot), Ldot=float(Ldot), flow=float(flow))

    A, f, act = build_masked_system_2_3(grid, active, node_eval, eps_drift=1e-12, check_inward=False)
    
    n_act = act.n_active
    D_shift = np.zeros(n_act)
    for p_act in range(n_act):
        node = unflatten(act.idx_full[p_act], grid)
        i, j = node
        tau, h, k, L = float(u_s["tau"][node]), float(u_s["h"][node]), float(grid.k[i]), float(grid.L[j])
        _, _, r_k = production_block(k, s, par)
        r_f = safe_rate(k, L, h, tau, r_k, par)
        pi_mc = market_clearing_risky_share(k, L, h)
        R_s = (1.0 - tau) * r_f + 0.5 * par.gamma * ((1.0 - tau)**2) * (par.sigma**2) * (pi_mc**2)
        D_shift[p_act] = (1.0 - par.gamma) * (R_s - omega_old[i, j])

    I = sp.eye(n_act, format="csr", dtype=float)
    D_mat = sp.diags(D_shift, format="csr")
    LHS = (par.rho + lam) * I - D_mat - A
    
    Psi_act = spla.spsolve(LHS.tocsc(), f)
    Psi_act = np.maximum(Psi_act, 1e-12) 
    
    Psi_full = embed_active_to_full(Psi_act, act, grid, anchor=np.nan)
    omega_new = omega_old.copy()
    valid = active & np.isfinite(Psi_full)
    omega_new[valid] = Psi_full[valid] ** (-1.0 / par.gamma)
    return omega_new

# -----------------------------
# 5. Core Solvers & Harness
# -----------------------------
def howard_inner_loop(grid, par, prim, *, lam, omega1, omega0, J1_init, J0_init, u1_init, u0_init, M1_init, M0_init, eta_policy=0.8, eps_drift=1e-12, m_inner_max=40, tol_policy=1e-7, verbose=False):
    M1, M0 = np.asarray(M1_init, dtype=bool).copy(), np.asarray(M0_init, dtype=bool).copy()
    M0 &= M1  
    J1, J0 = J1_init.copy(), J0_init.copy()
    u1, u0 = {k: np.asarray(v, dtype=float).copy() for k, v in u1_init.items()}, {k: np.asarray(v, dtype=float).copy() for k, v in u0_init.items()}

    for m in range(m_inner_max):
        def ne1(node):
            tau, h, T = float(u1["tau"][node]), float(u1["h"][node]), float(u1["T"][node])
            flow, kd, Ld = node_flow_and_drift(grid, par, prim, 1, node, tau, h, T, omega1)
            return NodeDriftFlow(kdot=float(kd), Ldot=float(Ld), flow=float(flow))
            
        A1, f1, act1 = build_masked_system_2_3(grid, M1, ne1, eps_drift=eps_drift, check_inward=False)
        J1_new = embed_active_to_full(solve_hjb_on_active(A1, f1, rho=par.rho), act1, grid, anchor=np.nan)

        def ne0(node):
            tau, h, T = float(u0["tau"][node]), float(u0["h"][node]), float(u0["T"][node])
            flow, kd, Ld = node_flow_and_drift(grid, par, prim, 0, node, tau, h, T, omega0)
            return NodeDriftFlow(kdot=float(kd), Ldot=float(Ld), flow=float(flow))
            
        A0, f0, act0 = build_masked_system_2_3(grid, M0, ne0, eps_drift=eps_drift, check_inward=False)
        J1_on_M0 = restrict_full_to_active(J1_new, act0, grid)
        J0_new = embed_active_to_full(solve_hjb_on_active(A0, f0 + lam * J1_on_M0, rho=par.rho, lam=lam), act0, grid, anchor=np.nan)

        u1_targ, M1_stable = improve_with_prune_closure(grid, par, prim, 1, J1_new, omega1, M1, eps_drift=eps_drift, max_passes=10)
        u0_targ, M0_stable = improve_with_prune_closure(grid, par, prim, 0, J0_new, omega0, M0, eps_drift=eps_drift, max_passes=10)
        M0_stable &= M1_stable

        u1_blend = blend_and_project_on_mask(u1, mask_policy(u1_targ, M1_stable), eta_policy, grid, par, prim, 1, M1_stable)
        u0_blend = blend_and_project_on_mask(u0, mask_policy(u0_targ, M0_stable), eta_policy, grid, par, prim, 0, M0_stable)

        kdot1, Ldot1 = _policy_drift_arrays_on_mask(grid, par, prim, 1, u1_blend, omega1, M1_stable, node_flow_and_drift=node_flow_and_drift)
        kdot0, Ldot0 = _policy_drift_arrays_on_mask(grid, par, prim, 0, u0_blend, omega0, M0_stable, node_flow_and_drift=node_flow_and_drift)
        ok1, ok0 = inward_one_cell(M1_stable, kdot1, Ldot1, eps=eps_drift), inward_one_cell(M0_stable, kdot0, Ldot0, eps=eps_drift)

        u1_next = mask_policy(_select_blend_or_snap_on_mask(u1, u1_blend, u1_targ, ok1, M1_stable), M1_stable)
        u0_next = mask_policy(_select_blend_or_snap_on_mask(u0, u0_blend, u0_targ, ok0, M0_stable), M0_stable)

        pol_diff = policy_supnorm(u1_next, u1, M1_stable) + policy_supnorm(u0_next, u0, M0_stable)
        if verbose: print(f"  [Howard] iter {m+1} pol_diff={pol_diff:.3e}")

        if pol_diff < tol_policy and np.array_equal(M1_stable, M1) and np.array_equal(M0_stable, M0):
            return J1_new, J0_new, u1_next, u0_next, M1_stable, M0_stable

        J1, J0, u1, u0, M1, M0 = J1_new, J0_new, u1_next, u0_next, M1_stable, (M0_stable & M1_stable)
    return J1, J0, u1, u0, M1, M0


def outer_loop_solver(
    grid, par, prim, *, lam, omega1_init, omega0_init, J1_init, J0_init, u1_init, u0_init, 
    M1_init, M0_init, Momega1, Momega0, primitive_feasible_set_fn, update_private_omega,
    zeta_omega=0.2, N_peel=5, eps_drift=1e-12, max_outer=200, tol_outer=1e-6,
    stable_window=3, min_mask_size=10, eta_policy=0.8, m_inner_max=40, tol_policy=1e-7, verbose=False,
):
    base_active = np.asarray(primitive_feasible_set_fn(grid, par, prim), dtype=bool) & np.asarray(grid.interior_mask, dtype=bool)

    # FIX: Initialize from the successfully pruned masks, not the raw grid!
    V1, V0 = M1_init.copy() & base_active, M0_init.copy() & base_active
    V0 &= V1
    M1, M0 = V1.copy(), V0.copy()

    omega1, omega0 = omega1_init.copy(), omega0_init.copy()
    J1, J0 = J1_init.copy(), J0_init.copy()
    u1 = {k: v.copy() for k,v in u1_init.items()}
    u0 = {k: v.copy() for k,v in u0_init.items()}

    history = []
    M1_prev, M0_prev, stable_count = None, None, 0
    Momega1_eff_prev, Momega0_eff_prev = None, None

    for n in range(max_outer):
        Momega1_eff, Momega0_eff = Momega1 & M1, Momega0 & M0

        omega1_new = update_private_omega(grid, par, prim, 1, omega1, u1, Momega1_eff)
        omega0_new = update_private_omega(grid, par, prim, 0, omega0, u0, Momega0_eff)

        omega1_half = (1.0 - zeta_omega) * omega1 + zeta_omega * omega1_new
        omega0_half = (1.0 - zeta_omega) * omega0 + zeta_omega * omega0_new

        omega1_ext = quarantine_fill_nearest(omega1_half, Momega1_eff)
        omega0_ext = quarantine_fill_nearest(omega0_half, Momega0_eff)

        V1 = viability_peel_warm(V1, base_active, grid, par, prim, omega1_ext, 1, static_feasible_node=static_feasible_node, drift_node=drift_node, N_peel=N_peel, eps_drift=eps_drift)
        V0 = viability_peel_warm(V0, base_active, grid, par, prim, omega0_ext, 0, static_feasible_node=static_feasible_node, drift_node=drift_node, N_peel=N_peel, eps_drift=eps_drift)
        V0 &= V1

        M1, M0 = V1.copy(), V0.copy()

        J1_new, J0_new, u1_new, u0_new, M1_new, M0_new = howard_inner_loop(
            grid, par, prim, lam=lam, omega1=omega1_ext, omega0=omega0_ext,
            J1_init=J1, J0_init=J0, u1_init=u1, u0_init=u0, M1_init=M1, M0_init=M0,
            eta_policy=eta_policy, eps_drift=eps_drift, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=False
        )

        core1, core0 = M1 & M1_new, M0 & M0_new
        core_om1 = Momega1_eff if Momega1_eff_prev is None else (Momega1_eff & Momega1_eff_prev)
        core_om0 = Momega0_eff if Momega0_eff_prev is None else (Momega0_eff & Momega0_eff_prev)

        d_omega = _core_max_norm(omega1_ext - omega1, core_om1, "d_omega1") + _core_max_norm(omega0_ext - omega0, core_om0, "d_omega0")
        d_J = _core_max_norm(J1_new - J1, core1, "d_J1") + _core_max_norm(J0_new - J0, core0, "d_J0")
        d_u = policy_supnorm(u1_new, u1, core1) + policy_supnorm(u0_new, u0, core0)

        history.append({"outer_iter": n, "d_omega": float(d_omega), "d_J": float(d_J), "d_u": float(d_u), "size_M1": int(M1_new.sum()), "size_M0": int(M0_new.sum())})
        if verbose: print(f"[Outer {n+1}/{max_outer}] d_omega={d_omega:.3e} d_J={d_J:.3e} d_u={d_u:.3e} |M1|={int(M1_new.sum())}")

        if (M1_prev is not None) and np.array_equal(M1_new, M1_prev) and np.array_equal(M0_new, M0_prev): stable_count += 1
        else: stable_count = 0
        
        M1_prev, M0_prev = M1_new.copy(), M0_new.copy()
        omega1, omega0 = omega1_ext, omega0_ext
        J1, J0, u1, u0 = J1_new, J0_new, u1_new, u0_new
        Momega1_eff_prev, Momega0_eff_prev = Momega1_eff.copy(), Momega0_eff.copy()

        V1 &= M1_new
        V0 &= (M0_new & V1)
        M1, M0 = V1.copy(), V0.copy()

        if (d_omega < tol_outer) and (d_J < tol_outer) and (d_u < tol_outer) and (stable_count >= stable_window):
            if verbose: print("[Outer] Converged.")
            break

    return {"omega1": omega1, "omega0": omega0, "J1": J1, "J0": J0, "u1": u1, "u0": u0, "M1": M1.copy(), "M0": M0.copy(), "history": history}


def initialize_policy_safe(grid, par, prim, s, omega, M_init, T_grid=None, eps_drift=1e-12, coarse_init=True):
    import copy
    prim_use = copy.copy(prim)
    if coarse_init:
        prim_use.tau_grid = np.linspace(prim.tau_min, prim.tau_max, 5)
        prim_use.h_grid = np.linspace(prim.h_min, prim.h_max, 5)
    
    J0 = np.zeros(grid.shape)
    u_targ, M_stable = improve_with_prune_closure(grid, par, prim_use, s, J0, omega, M_init, eps_drift=eps_drift, max_passes=25, T_grid=T_grid)
    
    if not np.any(M_stable):
        raise RuntimeError(
            f"Initialization Failed in Regime {s}: Mask collapsed completely!\n"
            f"Your boundaries cannot be trapped with primitive limits [T: {prim.T_min} to {prim.T_max}].\n"
            f"Please widen T_min and T_max (e.g. T_min=-2.0, T_max=4.0) in your setup cell."
        )
    return mask_policy(u_targ, M_stable), M_stable


def run_planner_frozen_omega(grid, par, prim, *, lam, omega_level=0.05, T_grid=None, eps_drift=1e-12, zeta_omega=0.5, N_peel=3, max_outer=10, tol_outer=1e-8, eta_policy=0.8, m_inner_max=25, tol_policy=1e-7, coarse_init=True, verbose=True):
    base_active = np.asarray(primitive_feasible_set_fallback(grid, par, prim), dtype=bool) & grid.interior_mask
    Momega1, Momega0 = make_owner_domains(base_active)
    omega1_init, omega0_init = np.full(grid.shape, float(omega_level)), np.full(grid.shape, float(omega_level))
    
    u1_init, M1_stable = initialize_policy_safe(grid, par, prim, 1, omega1_init, base_active.copy(), T_grid=T_grid, eps_drift=eps_drift, coarse_init=coarse_init)
    u0_init, M0_stable = initialize_policy_safe(grid, par, prim, 0, omega0_init, base_active.copy() & M1_stable, T_grid=T_grid, eps_drift=eps_drift, coarse_init=coarse_init)
    M0_stable &= M1_stable

    if verbose: print(f"\n[frozen ω] Starting outer loop |M1|={int(M1_stable.sum())} |M0|={int(M0_stable.sum())}")

    return outer_loop_solver(
        grid, par, prim, lam=lam, omega1_init=omega1_init, omega0_init=omega0_init,
        J1_init=np.zeros(grid.shape), J0_init=np.zeros(grid.shape), u1_init=u1_init, u0_init=u0_init,
        M1_init=M1_stable, M0_init=M0_stable, 
        Momega1=Momega1, Momega0=Momega0, primitive_feasible_set_fn=primitive_feasible_set_fallback,
        update_private_omega=update_private_omega_frozen, 
        zeta_omega=zeta_omega, N_peel=N_peel, 
        max_outer=max_outer, tol_outer=tol_outer,
        eta_policy=eta_policy, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=verbose
    )

def run_planner_active_omega(grid, par, prim, *, lam, omega_level=0.05, T_grid=None, eps_drift=1e-12, zeta_omega=0.5, N_peel=3, max_outer=10, tol_outer=1e-8, eta_policy=0.8, m_inner_max=25, tol_policy=1e-7, coarse_init=True, verbose=True):
    base_active = np.asarray(primitive_feasible_set_fallback(grid, par, prim), dtype=bool) & grid.interior_mask
    Momega1, Momega0 = make_owner_domains(base_active)
    omega1_init, omega0_init = np.full(grid.shape, float(omega_level)), np.full(grid.shape, float(omega_level))
    
    u1_init, M1_stable = initialize_policy_safe(grid, par, prim, 1, omega1_init, base_active.copy(), T_grid=T_grid, eps_drift=eps_drift, coarse_init=coarse_init)
    u0_init, M0_stable = initialize_policy_safe(grid, par, prim, 0, omega0_init, base_active.copy() & M1_stable, T_grid=T_grid, eps_drift=eps_drift, coarse_init=coarse_init)
    M0_stable &= M1_stable

    if verbose: print(f"\n[active ω] Starting outer loop |M1|={int(M1_stable.sum())} |M0|={int(M0_stable.sum())}")

    last_omega1_new = [omega1_init]
    def wrapped_update_omega(g, p, pr, s, omg_old, u, M_omg):
        if s == 1:
            omg1 = update_private_omega(g, p, pr, 1, omg_old, u, M_omg, lam=0.0, omega1_new=None)
            last_omega1_new[0] = omg1
            return omg1
        else:
            return update_private_omega(g, p, pr, 0, omg_old, u, M_omg, lam=lam, omega1_new=last_omega1_new[0])

    return outer_loop_solver(
        grid, par, prim, lam=lam, omega1_init=omega1_init, omega0_init=omega0_init,
        J1_init=np.zeros(grid.shape), J0_init=np.zeros(grid.shape), u1_init=u1_init, u0_init=u0_init,
        M1_init=M1_stable, M0_init=M0_stable, 
        Momega1=Momega1, Momega0=Momega0, primitive_feasible_set_fn=primitive_feasible_set_fallback,
        update_private_omega=wrapped_update_omega, 
        zeta_omega=zeta_omega, N_peel=N_peel, 
        max_outer=max_outer, tol_outer=tol_outer,
        eta_policy=eta_policy, m_inner_max=m_inner_max, tol_policy=tol_policy, verbose=verbose
    )


# In[ ]:


### ============================================================
if __name__ == "__main__":
    # Optional diagnostic dashboard
    # DIAGNOSTIC CHECK — Frozen Omega Integration Run & Plot
    # ============================================================
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    print("Setting up diagnostic grid and parameters...")

    # 1. Coarse grid for fast testing
    k_grid = np.linspace(0.5, 4.0, 15)
    L_grid = np.linspace(-0.5, 1.0, 15)
    grid_diag = Grid(k_grid, L_grid)

    # 2. Standard Macro Calibration
    par_diag = Par(
        rho=0.05, gamma=2.0, chi=0.6, 
        delta=0.05, g=0.02, sigma=0.15, 
        I0=0.3, I1=0.5
    )

    # 3. Policy Instrument Grids
    prim_diag = Prim(
        tau_grid=np.linspace(0.0, 0.4, 9),  # Planner can tax 0% to 40%
        h_grid=np.linspace(0.0, 2.0, 5),    # Government asset/debt scaling
        T_min=-0.2, T_max=1.0               # Transfer bounds
    )

    print("Firing up the Outer Loop Solver (Frozen Omega)...")
    start_time = time.time()

    # 4. Run the harness
    sol_diag = run_planner_frozen_omega(
        grid=grid_diag, 
        par=par_diag, 
        prim=prim_diag, 
        lam=0.3, 
        omega_level=0.05,
        zeta_omega=0.5, 
        N_peel=2, 
        max_outer=5,           # Keep iterations low for quick diagnostic
        tol_outer=1e-5,
        eta_policy=0.8, 
        m_inner_max=15, 
        tol_policy=1e-5,
        coarse_init=True,
        verbose=True
    )

    print(f"\nDiagnostic run completed in {time.time() - start_time:.1f} seconds.")

    # ---------------------------------------------------------
    # 5. Diagnostic Plotting Dashboard
    # ---------------------------------------------------------
    # Extract a 1D slice along the k-axis, keeping L closest to 0.0
    j_slice = np.argmin(np.abs(grid_diag.L)) 
    L_val = grid_diag.L[j_slice]
    k_valid = grid_diag.k

    # Extract Regime 1 (Post-Automation)
    M1_slice = sol_diag["M1"][:, j_slice]
    J1_slice = sol_diag["J1"][:, j_slice]
    tau1_slice = sol_diag["u1"]["tau"][:, j_slice]
    T1_slice = sol_diag["u1"]["T"][:, j_slice]

    # Mask out infeasible areas with NaN for clean plotting
    J1_plot = np.where(M1_slice, J1_slice, np.nan)
    tau1_plot = np.where(M1_slice, tau1_slice, np.nan)
    T1_plot = np.where(M1_slice, T1_slice, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Diagnostic Dashboard: Post-Automation Regime (Sliced at L ≈ {L_val:.2f})", fontsize=14, y=1.05)

    # Value Function Plot
    axes[0].plot(k_valid, J1_plot, 'b-o', lw=2)
    axes[0].set_title("Planner Value Function ($J_1$)")
    axes[0].set_xlabel("Physical Capital (k)")
    axes[0].set_ylabel("Utility")
    axes[0].grid(True, alpha=0.3)

    # Tax Policy Plot
    axes[1].plot(k_valid, tau1_plot, 'r-s', lw=2)
    axes[1].set_title("Optimal Capital Tax ($\\tau_1$)")
    axes[1].set_xlabel("Physical Capital (k)")
    axes[1].set_ylabel("Tax Rate")
    axes[1].set_ylim(prim_diag.tau_min - 0.05, prim_diag.tau_max + 0.05)
    axes[1].grid(True, alpha=0.3)

    # Transfer Policy Plot
    axes[2].plot(k_valid, T1_plot, 'g-^', lw=2)
    axes[2].set_title("Optimal Worker Transfer ($T_1$)")
    axes[2].set_xlabel("Physical Capital (k)")
    axes[2].set_ylabel("Transfer")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # In[ ]:




