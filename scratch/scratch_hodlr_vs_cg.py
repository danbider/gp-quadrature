"""
HODLR direct solver vs CG on efgpnd's mean system.

The A_mean operator in efgpnd is   A_mean = sigma^2 I + D T D   on R^M,
where T is an M x M symmetric Toeplitz matrix (T = F^* F) and D = diag(w)
with w_i = sqrt(h^d * khat(xi_i)) the spectral weights.  Current solver: CG
(see cg.py and efgpnd.create_A_mean).

Key reformulation that makes this HODLR-friendly:
    sigma^2 I + D T D = D (T + sigma^2 D^{-2}) D
    => let gamma = D beta.  Then (T + sigma^2 D^{-2}) gamma = D^{-1} r,
       and beta = gamma / D.
The inner operator  M := T + sigma^2 D^{-2}  is  Toeplitz + diagonal.
Since a diagonal doesn't alter off-diagonal rank, M inherits T's low
off-diagonal numerical rank.  For SE/Matern kernels these ranks are
10-30 at tol ~ 1e-8, so HODLR gives a true O(n log^2 n) direct
factorization.  The crucial feature for inference: the factorization
cost is independent of how nasty D gets (D's dynamic range can span
many orders of magnitude -- exactly the regime where CG slows down).

This script:
  (1) builds T and D to match efgpnd's 1D SE setup,
  (2) factorizes M = T + sigma^2 D^{-2} with a pure-numpy HODLR solver,
  (3) compares solve time + accuracy against CG (Jacobi-preconditioned)
      and a dense backslash baseline,
  (4) sweeps N and reports scalings.

Requires: numpy, scipy.  No pyhodlr-style dep.  Run via the user's
/Users/colecitrenbaum/myenv/bin/python.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.linalg as sla
from scipy.fft import fft, ifft


# ---------------------------------------------------------------------------
# HODLR direct solver (pure numpy, symmetric A, SVD compression).
# ---------------------------------------------------------------------------


@dataclass
class _HODLRStats:
    ranks: list         # list of (block_n, r1, r2) at each non-leaf node
    leaf_sizes: list    # leaf block sizes
    matvecs: int = 0    # total full-A matvecs consumed during construction
    level_ranks: dict = None  # level -> list of ranks seen at that level
    def bump(self, k):
        self.matvecs += k


class HODLR:
    """HODLR direct factorization for a symmetric n x n matrix A.

    Two constructors:
      - ``HODLR(A, ...)``: dense-input; uses deterministic SVD on each
        off-diagonal block.  O(n^2) top-level construction, exact rank.
      - ``HODLR.from_matvec(apply_fn, n, ...)``: black-box construction;
        only uses full-A matvecs (plus full rows at the leaves).
        Suitable for large n where forming A densely is too expensive.

    After construction, ``solve(b)`` is O(n log n).
    """

    def __init__(self, A: np.ndarray, leaf_size: int = 64, tol: float = 1e-10,
                 _stats: Optional[_HODLRStats] = None,
                 _depth: int = 0):
        assert A.ndim == 2 and A.shape[0] == A.shape[1]
        n = A.shape[0]
        self.n = n
        self.dtype = A.dtype
        self._depth = _depth
        if _stats is None:
            _stats = _HODLRStats([], [], 0, {})
        self._stats = _stats

        if n <= leaf_size:
            self.is_leaf = True
            self._stats.leaf_sizes.append(n)
            self.lu = sla.lu_factor(A)
            return

        self.is_leaf = False
        m = n // 2
        self.m = m

        A11 = A[:m, :m]
        A22 = A[m:, m:]
        A12 = A[:m, m:]
        A21 = A[m:, :m]

        # Compress off-diagonal blocks via truncated SVD.
        #   A12 = P @ Q.T   (P: m x r1, Q: (n-m) x r1)
        #   A21 = R @ S.T   (R: (n-m) x r2, S: m x r2)
        U, s, Vt = np.linalg.svd(A12, full_matrices=False)
        r1 = int(max(1, (s > tol * max(s[0], 1e-300)).sum()))
        self.P = U[:, :r1] * s[:r1]
        self.Q = Vt[:r1, :].T

        U, s, Vt = np.linalg.svd(A21, full_matrices=False)
        r2 = int(max(1, (s > tol * max(s[0], 1e-300)).sum()))
        self.R = U[:, :r2] * s[:r2]
        self.S = Vt[:r2, :].T

        self._stats.ranks.append((n, r1, r2))
        self._stats.level_ranks.setdefault(_depth, []).extend([r1, r2])

        # Recurse.
        self.top = HODLR(A11, leaf_size=leaf_size, tol=tol,
                         _stats=self._stats, _depth=_depth + 1)
        self.bot = HODLR(A22, leaf_size=leaf_size, tol=tol,
                         _stats=self._stats, _depth=_depth + 1)

        # Precompute D^{-1} U columns (multi-RHS solve against children).
        self.DP = self.top.solve(self.P)   # (m, r1)    = top^-1 P
        self.DR = self.bot.solve(self.R)   # (n-m, r2)  = bot^-1 R

        # Middle matrix  (I + V^T D^{-1} U)  where U has block-diag [P, 0; 0, R]
        # and V has anti-block [0, S; Q, 0], so V^T D^{-1} U =
        #     [ 0,            Q^T @ DR ]
        #     [ S^T @ DP,     0        ]
        r = r1 + r2
        Mmid = np.zeros((r, r), dtype=self.dtype)
        Mmid[:r1, r1:] = self.Q.T @ self.DR
        Mmid[r1:, :r1] = self.S.T @ self.DP
        self.r1 = r1
        self.r2 = r2
        self.mid_lu = sla.lu_factor(np.eye(r, dtype=self.dtype) + Mmid)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve  A x = b.  b may be (n,) or (n, k) for multi-RHS."""
        if self.is_leaf:
            return sla.lu_solve(self.lu, b)

        m = self.m
        b_top = b[:m]
        b_bot = b[m:]
        # D^{-1} b.
        z_top = self.top.solve(b_top)
        z_bot = self.bot.solve(b_bot)

        # V^T z   where V has block structure [[0, S], [Q, 0]].
        Vt_z = np.concatenate([self.Q.T @ z_bot, self.S.T @ z_top], axis=0)

        c = sla.lu_solve(self.mid_lu, Vt_z)

        Uc_top = self.DP @ c[: self.r1]
        Uc_bot = self.DR @ c[self.r1 :]
        return np.concatenate([z_top - Uc_top, z_bot - Uc_bot], axis=0)

    @property
    def max_rank(self) -> int:
        if not self._stats.ranks:
            return 0
        return max(max(r1, r2) for _, r1, r2 in self._stats.ranks)

    # -----------------------------------------------------------------
    # Black-box (matvec-only) constructor.
    # -----------------------------------------------------------------

    @classmethod
    def from_matvec(cls, apply_fn: Callable[[np.ndarray], np.ndarray],
                    n: int, *, leaf_size: int = 64, tol: float = 1e-8,
                    oversample: int = 10, rank_guess: int = 20,
                    rank_max: int = 200, rng: Optional[np.random.Generator] = None,
                    dtype=np.float64) -> "HODLR":
        """Construct a HODLR factorization using only full-A matvecs.

        ``apply_fn(v)`` applies A to a vector (or matrix, column-wise) of
        length n.  We assume A is symmetric.  Each off-diagonal block is
        compressed via randomized SVD with adaptive rank.  Leaf blocks
        are materialized with ``leaf_size`` full matvecs each.

        Stats: ``result._stats.matvecs`` reports the total number of
        full-A matvecs consumed during construction.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        stats = _HODLRStats(ranks=[], leaf_sizes=[], matvecs=0, level_ranks={})

        def _block_matvec(v: np.ndarray, row_slice: slice, col_slice: slice) -> np.ndarray:
            """Apply A[row_slice, col_slice] @ v via a full matvec.

            v shape: (col_len,) or (col_len, k).  Returns shape (row_len,)/(row_len, k).
            """
            col_len = col_slice.stop - col_slice.start
            one_d = (v.ndim == 1)
            if one_d:
                v = v[:, None]
            k = v.shape[1]
            V_full = np.zeros((n, k), dtype=dtype)
            V_full[col_slice, :] = v
            stats.matvecs += k
            Y_full = apply_fn(V_full)
            Y = Y_full[row_slice, :]
            return Y[:, 0] if one_d else Y

        def _compress_block(row_slice: slice, col_slice: slice) -> tuple:
            """Randomized SVD of A[row_slice, col_slice].  Returns (L, R, r)
            with block ≈ L @ R.T."""
            m = row_slice.stop - row_slice.start
            n_ = col_slice.stop - col_slice.start
            r = min(rank_guess, m, n_)
            while True:
                k = min(r + oversample, m, n_)
                Omega = rng.standard_normal((n_, k)).astype(dtype)
                Y = _block_matvec(Omega, row_slice, col_slice)    # (m, k)
                Q, _ = np.linalg.qr(Y)                             # (m, k)
                # B.T = A[rows, cols].T @ Q = A[cols, rows] @ Q (symmetric A)
                # But A[cols, rows] != A[rows, cols] in general; we access it
                # using the same zero-pad-matvec trick.
                BT = _block_matvec(Q, col_slice, row_slice)       # (n_, k) = A[col_slice, row_slice] @ Q
                # SVD of BT so block ≈ Q @ BT.T ≈ (Q @ Ub) * s * Vt
                Ub, s, Vt = np.linalg.svd(BT.T, full_matrices=False)
                # Adaptive rank selection based on singular value decay.
                if s[0] < 1e-300:
                    return (np.zeros((m, 1), dtype=dtype),
                            np.zeros((n_, 1), dtype=dtype), 1)
                eff = int(max(1, (s > tol * s[0]).sum()))
                # If we used all of k singular values as "significant", we may
                # have missed mass — double rank and retry.
                if eff >= k - 1 and r < min(rank_max, m, n_):
                    r = min(r * 2, rank_max, m, n_)
                    continue
                L = Q @ Ub[:, :eff]
                R = (Vt[:eff, :] * s[:eff, None]).T
                return L, R, eff

        def _leaf_dense(row_slice: slice, col_slice: slice) -> np.ndarray:
            """Materialize a (row_slice, col_slice) block by probing with
            standard basis vectors."""
            m = row_slice.stop - row_slice.start
            n_ = col_slice.stop - col_slice.start
            # Apply A to each basis vector of col_slice — one matvec per col.
            # Stack them to get the block.
            I = np.eye(n_, dtype=dtype)
            return _block_matvec(I, row_slice, col_slice)

        def _build(idx_slice: slice, depth: int) -> "HODLR":
            node = cls.__new__(cls)
            node._stats = stats
            node._depth = depth
            node.dtype = dtype
            a, b = idx_slice.start, idx_slice.stop
            sz = b - a
            node.n = sz
            if sz <= leaf_size:
                node.is_leaf = True
                block = _leaf_dense(idx_slice, idx_slice)
                stats.leaf_sizes.append(sz)
                node.lu = sla.lu_factor(block)
                return node
            node.is_leaf = False
            m = sz // 2
            node.m = m
            top_slice = slice(a, a + m)
            bot_slice = slice(a + m, b)
            # Compress off-diagonal blocks.
            P, Q, r1 = _compress_block(top_slice, bot_slice)   # A12
            R, S, r2 = _compress_block(bot_slice, top_slice)   # A21
            node.P = P; node.Q = Q
            node.R = R; node.S = S
            stats.ranks.append((sz, r1, r2))
            stats.level_ranks.setdefault(depth, []).extend([r1, r2])
            # Recurse.
            node.top = _build(top_slice, depth + 1)
            node.bot = _build(bot_slice, depth + 1)
            # Pre-factor middle matrix (same as dense path).
            node.DP = node.top.solve(node.P)
            node.DR = node.bot.solve(node.R)
            r = r1 + r2
            Mmid = np.zeros((r, r), dtype=dtype)
            Mmid[:r1, r1:] = node.Q.T @ node.DR
            Mmid[r1:, :r1] = node.S.T @ node.DP
            node.r1 = r1
            node.r2 = r2
            node.mid_lu = sla.lu_factor(np.eye(r, dtype=dtype) + Mmid)
            return node

        root = _build(slice(0, n), 0)
        root._stats = stats
        return root

    # -----------------------------------------------------------------
    # Peeled black-box constructor (symmetric A, one structured probe
    # per level instead of per-block).  Matvec count ~ L * (r+p) + leaf.
    # -----------------------------------------------------------------

    @classmethod
    def from_matvec_peeled(cls, apply_fn: Callable[[np.ndarray], np.ndarray],
                           n: int, *, leaf_size: int = 64, tol: float = 1e-8,
                           rank: int = 20, oversample: int = 10,
                           rng: Optional[np.random.Generator] = None,
                           dtype=np.float64) -> "HODLR":
        """Martinsson-style peeled construction for a SYMMETRIC A.

        At each level k = 0, 1, ..., L-1, apply A to a *single* structured
        probe of width r+p: Ω has random values on the "right half" of
        every super-node at that level, zeros on the "left halves".  Y = A
        Ω gives us the action of A[left_half, right_half] for every
        super-node at level k — but also the action of ancestor off-diag
        blocks we've already compressed.  We subtract those off (peeling).

        By symmetry, one probe per level is enough: A[top, bot] =
        A[bot, top]^T, so the two off-diag blocks at each super-node
        share U/V factors.

        Leaves are materialized by one batched matvec with identity
        columns restricted to each leaf (disjoint column supports across
        leaves, so one batched matvec of leaf_size cols materializes
        *all* leaf diagonal blocks at once).

        Total matvecs: L * (r + p) + leaf_size (+ one tiny check probe).
        """
        if rng is None:
            rng = np.random.default_rng(0)
        stats = _HODLRStats(ranks=[], leaf_sizes=[], matvecs=0, level_ranks={})

        L = max(1, int(np.ceil(np.log2(max(1, n / leaf_size)))))
        r = rank
        p = oversample

        # ----- Precompute super-node structure at each level -----
        # At level k, supernodes have size 2 * (n / 2^{k+1}) = n / 2^k.
        # To keep integer bookkeeping clean, we partition [0, n) greedily
        # using a recursive halving that matches the __init__ partition.
        def _partition(a: int, b: int, depth: int) -> list[tuple[int, int, int, int]]:
            """Return list of (a, mid, b, depth) for every non-leaf node at
            this depth and below, representing super-nodes to split."""
            sz = b - a
            if sz <= leaf_size:
                return []
            mid = a + sz // 2
            this_level = [(a, mid, b, depth)]
            return (this_level
                    + _partition(a, mid, depth + 1)
                    + _partition(mid, b, depth + 1))

        all_nodes = _partition(0, n, 0)
        nodes_by_depth: dict[int, list[tuple[int, int, int]]] = {}
        for a, m, b, d in all_nodes:
            nodes_by_depth.setdefault(d, []).append((a, m, b))

        # ----- Tree skeleton: build empty nodes matching the partition. -----
        def _make_empty_node(a: int, b: int, depth: int) -> "HODLR":
            node = cls.__new__(cls)
            node._stats = stats
            node._depth = depth
            node.dtype = dtype
            node.n = b - a
            node._a, node._b = a, b  # range for peeling
            sz = b - a
            if sz <= leaf_size:
                node.is_leaf = True
                stats.leaf_sizes.append(sz)
                node.lu = None  # filled in below
                return node
            node.is_leaf = False
            mid = a + sz // 2
            node.m = mid - a
            node.top = _make_empty_node(a, mid, depth + 1)
            node.bot = _make_empty_node(mid, b, depth + 1)
            # placeholders — filled in during level-k processing.
            node.P = None; node.Q = None; node.R = None; node.S = None
            node.r1 = 0; node.r2 = 0
            return node

        root = _make_empty_node(0, n, 0)

        # Helper: lookup the node by (a, b) range.
        by_range: dict[tuple[int, int], "HODLR"] = {}
        def _index(nd: "HODLR"):
            by_range[(nd._a, nd._b)] = nd
            if not nd.is_leaf:
                _index(nd.top); _index(nd.bot)
        _index(root)

        # ----- Apply the *partially-built* HODLR to a matrix, using only
        #       already-filled off-diagonal factors at depths <= cap.
        def _apply_partial(X: np.ndarray, depth_cap: int) -> np.ndarray:
            """Compute (sum over off-diag blocks at depths 0..depth_cap) X,
            using the currently stored low-rank factors.  Shape (n, k)."""
            Y = np.zeros_like(X)
            for d in range(depth_cap + 1):
                for (a, mid, b) in nodes_by_depth.get(d, []):
                    nd = by_range[(a, b)]
                    if nd.P is None or nd.Q is None:
                        continue
                    # Top -> bot contribution: Y[top] += A[top, bot] X[bot]
                    #                                 = P (Q^T X[bot])
                    Y[a:mid, :] += nd.P @ (nd.Q.T @ X[mid:b, :])
                    # Bot -> top:  by symmetry, A[bot, top] = A[top, bot]^T
                    #            = Q P^T.  So Y[bot] += Q (P^T X[top]).
                    Y[mid:b, :] += nd.Q @ (nd.P.T @ X[a:mid, :])
            return Y

        # ----- Per-level probing (two structured probes per level). -----
        k_width = r + p

        for depth in range(L):
            depth_nodes = nodes_by_depth.get(depth, [])
            if not depth_nodes:
                continue
            # Probe 1 (forward): Ω random on right halves of super-nodes.
            Omega = np.zeros((n, k_width), dtype=dtype)
            for (a, mid, b) in depth_nodes:
                Omega[mid:b, :] = rng.standard_normal((b - mid, k_width)).astype(dtype)
            stats.matvecs += k_width
            Y = apply_fn(Omega)
            if depth > 0:
                Y = Y - _apply_partial(Omega, depth - 1)
            # Extract range basis Qbasis[s] of A[left, right] for each supernode s.
            range_bases: dict[tuple[int, int], np.ndarray] = {}
            for (a, mid, b) in depth_nodes:
                Y_loc = Y[a:mid, :]                                # (mid-a, k_width)
                Qb, _ = np.linalg.qr(Y_loc)                         # (mid-a, k_width)
                range_bases[(a, b)] = Qb

            # Probe 2 (adjoint): place Qbasis into left halves;  A Ω' gives
            # Z[right-half] = A[right, left] Qbasis = A[left, right]^T Qbasis
            # (symmetry), so A[left, right] ≈ Qbasis @ Z[right-half]^T.
            Omega2 = np.zeros((n, k_width), dtype=dtype)
            for (a, mid, b) in depth_nodes:
                Omega2[a:mid, :] = range_bases[(a, b)]
            stats.matvecs += k_width
            Z = apply_fn(Omega2)
            if depth > 0:
                Z = Z - _apply_partial(Omega2, depth - 1)

            for (a, mid, b) in depth_nodes:
                Qb = range_bases[(a, b)]                            # (mid-a, k_width)
                Zblk = Z[mid:b, :]                                   # (b-mid, k_width)
                # A[a:mid, mid:b] ≈ Qb @ Zblk^T.  Compress to target tol via SVD.
                Ub, svals, Vt = np.linalg.svd(Zblk, full_matrices=False)
                if svals[0] < 1e-300:
                    r_eff = 1
                else:
                    r_eff = int(max(1, (svals > tol * svals[0]).sum()))
                r_eff = min(r_eff, k_width, mid - a, b - mid)
                # Q_factor (right) gets the singular values folded in;
                # P_factor (left) stays orthonormal times the right-SV rotation.
                L_factor = Qb @ Vt[:r_eff, :].T                      # (mid-a, r_eff)
                R_factor = Ub[:, :r_eff] * svals[:r_eff]              # (b-mid, r_eff)
                nd = by_range[(a, b)]
                nd.P = L_factor
                nd.Q = R_factor
                # By symmetry  A[mid:b, a:mid] = A[a:mid, mid:b]^T = R_factor @ L_factor.T.
                nd.R = R_factor
                nd.S = L_factor
                nd.r1 = r_eff
                nd.r2 = r_eff
                stats.ranks.append((b - a, r_eff, r_eff))
                stats.level_ranks.setdefault(depth, []).extend([r_eff, r_eff])

        # ----- Leaves: materialize all leaf diagonal blocks with a single
        #       batched matvec of leaf_size columns.
        def _leaves(nd: "HODLR"):
            if nd.is_leaf:
                yield nd
            else:
                yield from _leaves(nd.top)
                yield from _leaves(nd.bot)

        leaves = list(_leaves(root))
        # All leaves have disjoint column supports. Place identity columns
        # into each leaf's columns simultaneously.
        max_leaf = max(lf.n for lf in leaves)
        E = np.zeros((n, max_leaf), dtype=dtype)
        for lf in leaves:
            cols = np.arange(lf.n)
            E[lf._a + cols, cols] = 1.0
        stats.matvecs += max_leaf
        Y = apply_fn(E)
        # Peel off all non-leaf off-diag contributions.
        Y = Y - _apply_partial(E, L - 1)
        for lf in leaves:
            cols = np.arange(lf.n)
            block = Y[lf._a + cols[:, None], cols[None, :]]   # (lf.n, lf.n)
            lf.lu = sla.lu_factor(block)

        # ----- Pre-factor internal nodes' middle matrices (DP, DR, mid_lu).
        #       Must be done bottom-up so children are factored first.
        def _prefactor(nd: "HODLR"):
            if nd.is_leaf:
                return
            _prefactor(nd.top)
            _prefactor(nd.bot)
            nd.DP = nd.top.solve(nd.P)
            nd.DR = nd.bot.solve(nd.R)
            rr = nd.r1 + nd.r2
            Mmid = np.zeros((rr, rr), dtype=dtype)
            Mmid[:nd.r1, nd.r1:] = nd.Q.T @ nd.DR
            Mmid[nd.r1:, :nd.r1] = nd.S.T @ nd.DP
            nd.mid_lu = sla.lu_factor(np.eye(rr, dtype=dtype) + Mmid)

        _prefactor(root)
        root._stats = stats
        return root


# ---------------------------------------------------------------------------
# Toeplitz mat-vec (symmetric, real) via FFT — matches efgpnd's T = F^* F
# up to the outer D in create_Gv; here we isolate the *Toeplitz itself*.
# ---------------------------------------------------------------------------


def toeplitz_matvec_fft(col: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply symmetric Toeplitz T (first column=`col`) to x.

    Supports x of shape (n,) or (n, k).  Uses circulant embedding, one FFT
    of size 2n applied along axis 0.
    """
    n = col.shape[0]
    c = np.concatenate([col, [0.0], col[-1:0:-1]])
    c_fft = np.fft.fft(c)                        # (2n,)
    if x.ndim == 1:
        y = np.fft.ifft(c_fft * np.fft.fft(x, n=2 * n))[:n].real
        return y
    else:
        x_fft = np.fft.fft(x, n=2 * n, axis=0)   # (2n, k)
        y = np.fft.ifft(c_fft[:, None] * x_fft, axis=0)[:n, :].real
        return y


def build_toeplitz_dense(col: np.ndarray) -> np.ndarray:
    return sla.toeplitz(col)


# ---------------------------------------------------------------------------
# CG (reference; matches cg.py semantics).
# ---------------------------------------------------------------------------


def cg(A_apply: Callable[[np.ndarray], np.ndarray], b: np.ndarray,
       tol: float = 1e-6, max_iter: Optional[int] = None,
       M_inv: Optional[Callable[[np.ndarray], np.ndarray]] = None
       ) -> tuple[np.ndarray, int, float]:
    n = b.shape[0]
    if max_iter is None:
        max_iter = 2 * n
    x = np.zeros_like(b)
    r = b - A_apply(x)
    z = M_inv(r) if M_inv is not None else r
    p = z.copy()
    rz = float(np.dot(r, z))
    bn = np.linalg.norm(b) + 1e-300
    for it in range(1, max_iter + 1):
        Ap = A_apply(p)
        pAp = float(np.dot(p, Ap)) + 1e-300
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        relres = np.linalg.norm(r) / bn
        if relres < tol:
            return x, it, relres
        z = M_inv(r) if M_inv is not None else r
        rz_new = float(np.dot(r, z))
        p = z + (rz_new / (rz + 1e-300)) * p
        rz = rz_new
    return x, max_iter, relres


# ---------------------------------------------------------------------------
# Build the efgpnd 1D mean system:  A_mean = sigma^2 I + D T D.
#
# We follow efgpnd: pick an equispaced Fourier grid of spacing h with M
# nodes, set D = diag(sqrt(h * khat(xi))).  Here we just construct T as
# a real symmetric Toeplitz with the SE real-space kernel on an arbitrary
# 1D grid (this matches F^* F for N training points drawn uniformly, up
# to weighting; for the solver benchmark the exact source of T is
# immaterial -- what matters is it's Toeplitz with low off-diagonal rank).
# ---------------------------------------------------------------------------


def se_khat(xi: np.ndarray, ell: float, var: float) -> np.ndarray:
    """1D squared-exponential spectral density (Fourier of k(r)=var*exp(-r^2/(2 ell^2)))."""
    return var * ell * math.sqrt(2 * math.pi) * np.exp(-0.5 * (2 * math.pi * xi * ell) ** 2)


def build_T_D(M: int, ell: float, var: float, sigma2: float,
              box_scale: float = 6.0, khat_floor: float = 1e-14
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_col, d_vec, xi_grid) where T_col is the Toeplitz first column.

    T is built as a Toeplitz with entries T_ij = sum_k exp(2*pi*i*(i-j)*h*xi_k),
    but for this bench we just use the real-space SE kernel T_ij = k(|i-j|*dx)
    -- this is Toeplitz, SPD, and has the low-rank off-diagonal structure that
    HODLR exploits, which is exactly the situation in efgpnd.
    """
    # Fourier grid covering roughly box_scale / ell in each direction.
    xi_max = box_scale / (2 * math.pi * ell)
    xis = np.linspace(-xi_max, xi_max, M)
    h = xis[1] - xis[0]

    d_raw = np.sqrt(np.maximum(h * se_khat(xis, ell, var), khat_floor))

    # Toeplitz: use smooth SE kernel in *index space*.  Pick dx so the
    # kernel is well-resolved on the grid.  The actual efgpnd T is
    # F^*F = prefix-sum of exponentials; the important structural fact
    # for HODLR is that T is Toeplitz with a smooth generator, which
    # real-space SE satisfies.
    dx = ell / 8.0
    r = np.arange(M) * dx
    T_col = var * np.exp(-0.5 * (r / ell) ** 2)
    return T_col, d_raw, xis


# ---------------------------------------------------------------------------
# Benchmark driver.
# ---------------------------------------------------------------------------


def make_A_mean_apply(T_col: np.ndarray, d: np.ndarray, sigma2: float):
    def apply(x: np.ndarray) -> np.ndarray:
        return sigma2 * x + d * toeplitz_matvec_fft(T_col, d * x)
    return apply


def make_M_apply(T_col: np.ndarray, d: np.ndarray, sigma2: float):
    """Apply  M = T + sigma^2 D^{-2}  — the HODLR-target operator."""
    d2_inv = 1.0 / (d * d)
    def apply(x: np.ndarray) -> np.ndarray:
        Tx = toeplitz_matvec_fft(T_col, x)
        if x.ndim == 1:
            return Tx + sigma2 * d2_inv * x
        else:
            return Tx + sigma2 * d2_inv[:, None] * x
    return apply


def jacobi_precond(d: np.ndarray, T_col: np.ndarray, sigma2: float):
    # Diag(D T D) = d_i^2 * T_{ii} = d_i^2 * T_col[0].
    diag = d * d * T_col[0] + sigma2
    return lambda v: v / diag


def run_one(M: int, ell: float, var: float, sigma2: float, *,
            cg_tol: float = 1e-8, hodlr_tol: float = 1e-8,
            leaf_size: int = 64, rng: np.random.Generator | None = None,
            do_dense: bool = True, do_hodlr_dense: bool = False,
            do_hodlr_matvec: bool = True,
            do_hodlr_peeled: bool = True) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)

    T_col, d, xis = build_T_D(M, ell, var, sigma2)
    d_range = (float(d.max()) / float(d.min()))

    # rhs for the *original* A_mean system.
    r = rng.standard_normal(M).astype(np.float64)

    # --- dense reference ----
    dense_time = None
    x_ref = None
    if do_dense:
        t0 = time.perf_counter()
        T_dense = build_toeplitz_dense(T_col)
        A = sigma2 * np.eye(M) + d[:, None] * T_dense * d[None, :]
        x_ref = np.linalg.solve(A, r)
        dense_time = time.perf_counter() - t0

    # --- CG (current efgpnd pipeline) ----
    A_apply = make_A_mean_apply(T_col, d, sigma2)
    M_inv = jacobi_precond(d, T_col, sigma2)
    t0 = time.perf_counter()
    x_cg_pre, iters_pre, _ = cg(A_apply, r, tol=cg_tol, M_inv=M_inv)
    cg_pre_time = time.perf_counter() - t0

    # --- HODLR on reformulated M = T + sigma^2 D^{-2}, rhs = r/d, then x = gamma/d ----
    M_matvec = make_M_apply(T_col, d, sigma2)
    result = {
        "M": M, "d_dynamic_range": d_range,
        "dense_time": dense_time,
        "cg_pre_time": cg_pre_time, "cg_pre_iters": iters_pre,
    }

    def _finish_hodlr(H: "HODLR", label: str, build_time: float,
                      extra_build_overhead: float = 0.0):
        rhs_h = r / d
        t0 = time.perf_counter()
        gamma = H.solve(rhs_h)
        x = gamma / d
        solve_time = time.perf_counter() - t0
        if x_ref is not None:
            err = float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
        else:
            err = float("nan")
        result[f"{label}_factor_time"] = build_time + extra_build_overhead
        result[f"{label}_solve_time"] = solve_time
        result[f"{label}_max_rank"] = H.max_rank
        result[f"{label}_err"] = err
        result[f"{label}_level_ranks"] = dict(H._stats.level_ranks)
        result[f"{label}_matvecs"] = H._stats.matvecs

    if do_hodlr_dense:
        t0 = time.perf_counter()
        T_dense_local = build_toeplitz_dense(T_col)
        Mat = T_dense_local + np.diag(sigma2 / (d * d))
        build_dense_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        H_dense = HODLR(Mat, leaf_size=leaf_size, tol=hodlr_tol)
        _finish_hodlr(H_dense, "hodlr_dense",
                      time.perf_counter() - t0,
                      extra_build_overhead=build_dense_t)
        result["build_dense_time"] = build_dense_t

    if do_hodlr_matvec:
        t0 = time.perf_counter()
        H_mv = HODLR.from_matvec(M_matvec, M,
                                  leaf_size=leaf_size, tol=hodlr_tol,
                                  rank_guess=20, rank_max=80,
                                  rng=rng)
        _finish_hodlr(H_mv, "hodlr_mv", time.perf_counter() - t0)

    if do_hodlr_peeled:
        t0 = time.perf_counter()
        H_pl = HODLR.from_matvec_peeled(M_matvec, M,
                                         leaf_size=leaf_size, tol=hodlr_tol,
                                         rank=20, oversample=10, rng=rng)
        _finish_hodlr(H_pl, "hodlr_peel", time.perf_counter() - t0)

    return result


def _fmt_level_ranks(level_ranks: dict) -> str:
    if not level_ranks:
        return "-"
    lines = []
    for lvl in sorted(level_ranks.keys()):
        ranks = level_ranks[lvl]
        lines.append(f"L{lvl}:max={max(ranks)},median={int(np.median(ranks))}")
    return "  ".join(lines)


def main():
    print("=" * 78)
    print("HODLR  (solve T + sigma^2/D^2 gamma = r/D, x = gamma/D)")
    print("vs  CG  on A_mean = sigma^2 I + D T D")
    print("=" * 78)

    rng = np.random.default_rng(0)

    # -------- (1) n = 1024 sanity: dense + HODLR-dense + HODLR-matvec + CG. --------
    print("\n### 1. n=1024 sanity check (dense + HODLR-dense + HODLR-matvec + CG) ###\n")
    res0 = run_one(M=1024, ell=0.1, var=1.0, sigma2=1e-2, rng=rng,
                   do_dense=True, do_hodlr_dense=True, do_hodlr_matvec=True)
    print(f"  n = {res0['M']},  D dynamic range = {res0['d_dynamic_range']:.2e}")
    print(f"  dense backslash    time = {res0['dense_time']:.3f} s")
    print(f"  CG (Jacobi)        time = {res0['cg_pre_time']:.3f} s, iters = {res0['cg_pre_iters']}")
    print(f"  HODLR (dense SVD)  fact = {res0['hodlr_dense_factor_time']:.3f} s, "
          f"solve = {res0['hodlr_dense_solve_time']:.4f} s, "
          f"err = {res0['hodlr_dense_err']:.2e},  max_rank = {res0['hodlr_dense_max_rank']}")
    print(f"  HODLR (matvec)     fact = {res0['hodlr_mv_factor_time']:.3f} s, "
          f"solve = {res0['hodlr_mv_solve_time']:.4f} s, "
          f"err = {res0['hodlr_mv_err']:.2e},  max_rank = {res0['hodlr_mv_max_rank']},  "
          f"full-A matvecs consumed = {res0['hodlr_mv_matvecs']}")
    print(f"  HODLR-matvec per-level ranks:  {_fmt_level_ranks(res0['hodlr_mv_level_ranks'])}")
    print(f"  HODLR (peeled)     fact = {res0['hodlr_peel_factor_time']:.3f} s, "
          f"solve = {res0['hodlr_peel_solve_time']:.4f} s, "
          f"err = {res0['hodlr_peel_err']:.2e},  max_rank = {res0['hodlr_peel_max_rank']},  "
          f"full-A matvecs consumed = {res0['hodlr_peel_matvecs']}")
    print(f"  HODLR-peeled per-level ranks:  {_fmt_level_ranks(res0['hodlr_peel_level_ranks'])}")

    # -------- (2) Sweep N, matvec-only HODLR — peeled vs unpeeled vs Jacobi-CG. --------
    print("\n### 2. N sweep: peeled HODLR vs unpeeled HODLR vs Jacobi-preconditioned CG ###")
    print("    (wall time and total full-A matvecs)\n")
    hdr = (f"{'N':>6s} {'drange':>9s} | "
           f"{'CG time':>8s} {'CG mvs':>8s} | "
           f"{'unpeel time':>11s} {'unpeel mvs':>10s} {'err':>8s} | "
           f"{'peel time':>9s} {'peel mvs':>8s} {'err':>8s} {'rank':>4s}")
    print(hdr)
    print("-" * len(hdr))
    for M in [1024, 2048, 4096, 8192, 16384]:
        do_dense = (M <= 4096)
        res = run_one(M=M, ell=0.1, var=1.0, sigma2=1e-2, rng=rng,
                      do_dense=do_dense, do_hodlr_dense=False,
                      do_hodlr_matvec=True, do_hodlr_peeled=True)
        err_un = (f"{res['hodlr_mv_err']:.1e}" if not math.isnan(res['hodlr_mv_err']) else "--")
        err_pl = (f"{res['hodlr_peel_err']:.1e}" if not math.isnan(res['hodlr_peel_err']) else "--")
        # CG matvec count = iters (1 matvec / iter).
        print(f"{res['M']:>6d} {res['d_dynamic_range']:>9.1e} | "
              f"{res['cg_pre_time']:>7.3f}s {res['cg_pre_iters']:>8d} | "
              f"{res['hodlr_mv_factor_time']:>10.3f}s {res['hodlr_mv_matvecs']:>10d} {err_un:>8s} | "
              f"{res['hodlr_peel_factor_time']:>8.3f}s {res['hodlr_peel_matvecs']:>8d} {err_pl:>8s} "
              f"{res['hodlr_peel_max_rank']:>4d}")

    # -------- (3) D dynamic range sweep at fixed N: peeled HODLR vs Jacobi-CG. --------
    print("\n### 3. D-dynamic-range sweep (N=4096): peeled HODLR vs Jacobi-CG ###\n")
    hdr = (f"{'sigma^2':>9s} {'drange':>9s} | "
           f"{'CG time':>8s} {'CG mvs':>7s} | "
           f"{'peel time':>9s} {'peel mvs':>8s} {'rank':>4s} {'err':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for sigma2 in [1.0, 1e-2, 1e-4, 1e-6, 1e-8]:
        res = run_one(M=4096, ell=0.1, var=1.0, sigma2=sigma2, rng=rng,
                      do_dense=True, do_hodlr_dense=False,
                      do_hodlr_matvec=False, do_hodlr_peeled=True)
        err_pl = (f"{res['hodlr_peel_err']:.1e}" if not math.isnan(res['hodlr_peel_err']) else "--")
        print(f"{sigma2:>9.1e} {res['d_dynamic_range']:>9.1e} | "
              f"{res['cg_pre_time']:>7.3f}s {res['cg_pre_iters']:>7d} | "
              f"{res['hodlr_peel_factor_time']:>8.3f}s {res['hodlr_peel_matvecs']:>8d} "
              f"{res['hodlr_peel_max_rank']:>4d} {err_pl:>8s}")

    # -------- (4) Gradient-style cost: multiple RHS, fixed D ("inner loop"). --
    # The EFGP gradient solves A_mean against  1 + Hk*Tp + Tp  right-hand
    # sides (one mean solve + Hk kernel-hyper probes + Tp noise probes) at
    # *fixed* ws (i.e. fixed D).  HODLR factorizes once, amortizes across all
    # RHSs.  CG pays iters-per-RHS (batched, but still proportional to
    # iters * num_rhs matvecs).
    print("\n### 4. Gradient-cost model (fixed D, many RHS) ###\n")
    print("  We assume:  1 mean-solve  +  Tp=10 probes per (K kernel hypers + 1 noise)")
    print("             => num_rhs = 1 + (K+1) * Tp   RHS against A_mean.")
    print("  HODLR matvec count = (factorization matvecs) + 0 per solve.")
    print("  CG matvec count    = num_rhs * iters_per_solve.\n")

    hdr = (f"{'N':>6s} {'K':>3s} {'Tp':>3s} {'num_rhs':>8s} {'CGiter':>7s} "
           f"{'CG_mvs':>10s} {'CG_time':>8s} {'H_mvs':>8s} {'H_fact_t':>10s} "
           f"{'H_solve_t':>10s} {'speedup':>7s}")
    print(hdr)
    print("-" * len(hdr))

    for N in [1024, 2048, 4096, 8192, 16384]:
        T_col, d, _ = build_T_D(N, 0.1, 1.0, 1e-2)
        A_apply = make_A_mean_apply(T_col, d, 1e-2)
        M_inv = jacobi_precond(d, T_col, 1e-2)
        M_op = make_M_apply(T_col, d, 1e-2)
        K = 2   # kernel hypers (lengthscale, variance)
        Tp = 10  # trace probes
        num_rhs = 1 + (K + 1) * Tp
        B = rng.standard_normal((num_rhs, N)).astype(np.float64)

        # Jacobi-preconditioned CG, matching efgpnd/cg.py semantics.
        t0 = time.perf_counter()
        total_iters = 0
        for i in range(num_rhs):
            _, it, _ = cg(A_apply, B[i], tol=1e-8, M_inv=M_inv)
            total_iters += it
        cg_time = time.perf_counter() - t0

        # Peeled HODLR: factorize once, then many solves (O(N log N) per solve).
        t0 = time.perf_counter()
        H = HODLR.from_matvec_peeled(M_op, N, leaf_size=64, tol=1e-8,
                                      rank=20, oversample=10, rng=rng)
        h_fact = time.perf_counter() - t0
        h_fact_mvs = H._stats.matvecs
        t0 = time.perf_counter()
        for i in range(num_rhs):
            rhs_i = B[i] / d
            gamma_i = H.solve(rhs_i)
            _ = gamma_i / d
        h_solve = time.perf_counter() - t0

        avg_iter = total_iters / num_rhs
        cg_mvs = total_iters  # one A_apply per CG iter (summed over all RHS)
        speedup = cg_time / (h_fact + h_solve)
        print(f"{N:>6d} {K:>3d} {Tp:>3d} {num_rhs:>8d} "
              f"{avg_iter:>7.1f} {cg_mvs:>10d} {cg_time:>7.3f}s  "
              f"{h_fact_mvs:>8d} {h_fact:>9.3f}s  {h_solve:>9.4f}s "
              f"{speedup:>6.2f}x")


if __name__ == "__main__":
    main()
