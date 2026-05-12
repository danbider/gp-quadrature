## 1. The SING-GP step we need to replace

The step to sub out is `SparseGP.update_dynamics_params`. In the repo, this computes the GP drift posterior (q(u)) by explicitly constructing three sparse-GP expectation sums:

[
\sum_i \Delta_i,\mathbb E_{q(x_i)}[k_{z x_i}k_{x_i z}],
]

[
\sum_i \mathbb E_{q(x_i)}[k_{z x_i}](m_{i+1}-m_i-\Delta_i Bv_i),
]

[
\sum_i \mathbb E_{q(x_i)}\left[\frac{d k_{z x}}{dx}\right](S_{i,i+1}-S_i).
]

In code, the first term is built as `E_KzxKxz_on_grid` with shape `(T-1, M, M)` and then summed over time; the second uses `E_Kxz_on_grid`; the third uses `E_dKzxdx_on_grid`. The posterior update then solves with `Kzz + sigma^{-2} int_E_KzxKxz`. ([GitHub][1])

This is fine for a small inducing grid. But if we replace inducing points with (M=m^d) Fourier features, explicitly materializing the analogue of

[
\mathbb E[\phi(x_i)\phi(x_i)^*]
]

for every (i) would be (O(nM^2)) to build or at least (O(nM)) if we loop pointwise over features. That is exactly what we must avoid.

The SING paper’s Eq. 38 is the mathematical version of this same update: the posterior covariance and mean of (q(u)) depend on those three summed expectations, and the dependence on time appears only through sums over time steps. 

---

## 2. The target EFGP representation

Replace inducing variables (u) with whitened Fourier weights. For drift coordinate (r),

[
f_r(x) \approx \phi(x)^*w_r,
\qquad
w_r\sim \mathcal N(0,I),
]

where

[
\phi(x)=D_\theta F_x,
]

[
F_x[k] = \exp(2\pi i, x^\top \xi_k),
]

and (D_\theta) is diagonal with square-root spectral weights. The EFGP draft uses exactly this structure: (\Phi = FD), and when the frequencies are equispaced, (F^*\Delta F) is Toeplitz/BTTB for any diagonal (\Delta), giving (O(m^d\log m^d)) matvecs and only linear-in-(n) setup. 

For now assume diagonal diffusion,

[
\Sigma = \operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2),
]

or shared scalar diffusion (\sigma^2 I). Full (\Sigma) can be handled later, but it couples output dimensions.

---

## 3. The CAVI expression we need for (q(f))

Index valid transitions by (a). For example (a=(\ell,i)), with latent marginal

[
q(x_a)=\mathcal N(m_a,S_a),
]

cross-covariance correction

[
C_a := S_{a,a+1}-S_a,
]

time step (\Delta_a), and input-corrected increment

[
d_a := m_{a+1}-m_a-\Delta_a Bv_a.
]

The sparse-GP CAVI update becomes, in Fourier-weight form,

[
q(w_r)=\mathcal N(\mu_r,Q_r),
]

with

[
Q_r^{-1}
========

# A_r

I
+
\frac{1}{\sigma_r^2}
\sum_a
\Delta_a,
\mathbb E_{q(x_a)}
[
\phi(x_a)\phi(x_a)^*
],
]

and

[
A_r\mu_r
========

h_r,
]

where

[
h_r
===

\frac{1}{\sigma_r^2}
\left[
\sum_a
\mathbb E_{q(x_a)}[\phi(x_a)], d_{a,r}
+
\sum_a
\mathbb E_{q(x_a)}[\nabla_x\phi(x_a)]^\top C_{a,:,r}
\right].
]

This is the exact object to replace the inducing-point update with.

The crucial point: **all expectations are single-time expectations under (q(x_a))**. We do not need sampled pairs ((x_i,x_{i+1})) for this update. The temporal coupling enters through the already available (C_a=S_{a,a+1}-S_a).

---

## 4. Monte Carlo version that restores the EFGP Toeplitz structure

Draw marginal samples

[
x_a^{(s)}\sim q(x_a),
\qquad s=1,\ldots,S.
]

Then

[
\mathbb E[\phi(x_a)\phi(x_a)^*]
\approx
\frac1S\sum_{s=1}^S
\phi(x_a^{(s)})\phi(x_a^{(s)})^*.
]

So

[
A_r
\approx
I
+
D_\theta
\left[
F_X^* W_r F_X
\right]
D_\theta,
]

where (X={x_a^{(s)}}_{a,s}), and

[
(W_r)_{a,s}
===========

\frac{\Delta_a}{S\sigma_r^2}.
]

The middle matrix has entries

[
(F_X^*W_rF_X)_{k\ell}
=====================

\sum_{a,s}
\frac{\Delta_a}{S\sigma_r^2}
\exp\left(2\pi i, (x_a^{(s)})^\top(\xi_\ell-\xi_k)\right).
]

Because this depends only on the frequency difference (\xi_\ell-\xi_k), it is Toeplitz/BTTB on the equispaced frequency grid. So we never form (F_X) or (A_r). We only implement

[
v\mapsto A_r v
==============

v+
D_\theta
\left[
F_X^*W_rF_X
\right]
D_\theta v,
]

where the bracketed Toeplitz/BTTB operator is applied by FFT.

---

## 5. RHS construction with NUFFTs

The first RHS term is

[
h_{1,r}
=======

\frac{1}{\sigma_r^2}
\sum_a
\mathbb E[\phi(x_a)]d_{a,r}.
]

Monte Carlo gives

[
h_{1,r}
\approx
\frac{1}{\sigma_r^2}
\sum_{a,s}
\frac1S
\phi(x_a^{(s)})d_{a,r}.
]

Since (\phi=D_\theta F),

[
h_{1,r}
=======

D_\theta F_X^* a_r,
]

where

[
(a_r)_{a,s}
===========

\frac{d_{a,r}}{S\sigma_r^2}.
]

So this is one type-1 NUFFT per output dimension, or one batched type-1 NUFFT for all (r).

The derivative correction uses

[
\nabla_x \phi_k(x)
==================

(2\pi i,\xi_k)\phi_k(x).
]

Thus

[
h_{2,r}
=======

\frac{1}{\sigma_r^2}
\sum_a
\mathbb E[\nabla_x\phi(x_a)]^\top C_{a,:,r}
]

becomes

[
h_{2,r}
\approx
\sum_{j=1}^{D_{\text{lat}}}
(2\pi i,\xi_j)
\odot
D_\theta F_X^* c_{j,r},
]

where

[
(c_{j,r})_{a,s}
===============

\frac{C_{a,j,r}}{S\sigma_r^2}.
]

So the derivative correction is also just type-1 NUFFTs plus frequency multiplication. No explicit (N\times M) feature matrix.

---

## 6. What happens to (q(x)) once (q(f)) is updated?

Given

[
q(w_r)=\mathcal N(\mu_r,Q_r),
\qquad Q_r=A_r^{-1},
]

the SING latent update needs the drift moments. The current repo’s generic SDE interface is organized around exactly these primitives: `f`, `ff`, and `dfdx`, which compute (\mathbb E[f(x_i)]), (\mathbb E[|f(x_i)|^2]), and (\mathbb E[J_f(x_i)]) under the current (q(x_i)). The base implementation uses Gaussian integration, with Monte Carlo as the default when no expectation object is supplied. ([GitHub][1])

For EFGP, we should implement these **batched**, not one (x_i) at a time.

Mean drift:

[
\bar f_r(x)
===========

# \mathbb E_{q(f)}[f_r(x)]

\phi(x)^*\mu_r.
]

Jacobian:

[
\frac{\partial \bar f_r(x)}{\partial x_j}
=========================================

\phi(x)^*
\left[
(2\pi i,\xi_j)\odot \mu_r
\right].
]

Second moment:

[
\mathbb E_{q(f)}[f_r(x)^2]
==========================

\bar f_r(x)^2
+
\phi(x)^*Q_r\phi(x).
]

The posterior-variance term

[
v_r(X)
======

\operatorname{diag}(\Phi_X Q_r\Phi_X^*)
]

should be estimated by Hutchinson:

[
v_r(X)
\approx
\frac1J
\sum_{j=1}^J
z_j\odot
\Phi_X Q_r\Phi_X^*z_j.
]

Each probe requires:

[
z_j
\rightarrow
\Phi_X^*z_j
\rightarrow
A_r^{-1}\Phi_X^*z_j
\rightarrow
\Phi_X A_r^{-1}\Phi_X^*z_j.
]

That is one type-1 NUFFT, one CG solve using the same (A_r)-matvec, and one type-2 NUFFT. This mirrors the EFGP draft’s diagonal-covariance estimator: do not compute diagonal variances directly; estimate them using matrix-free covariance applies. 

---

## 7. E-step with EFGP: algorithm block

**Inputs:** current trajectory posterior (q(x_{0:T})), fixed kernel hyperparameters (\theta), fixed diffusion (\Sigma), fixed emissions, optional inputs (v_i), Fourier grid ({\xi_k}*{k=1}^M), spectral weights (D*\theta), number of MC samples (S), number of Hutchinson probes (J).

**Output:** updated (q(f)) and updated (q(x_{0:T})).

1. **Extract SING marginal moments.**
   From the current Gaussian Markov posterior, get
   [
   m_a,\quad S_a,\quad C_a=S_{a,a+1}-S_a
   ]
   for every valid transition (a). Also compute
   [
   d_a=m_{a+1}-m_a-\Delta_a Bv_a.
   ]

2. **Sample uncertain latent inputs for the GP drift update.**
   For each valid transition (a), draw
   [
   x_a^{(s)}\sim \mathcal N(m_a,S_a),
   \qquad s=1,\ldots,S.
   ]
   Use marginal samples only; sampled pairs are unnecessary for the Eq. 38-style update.

3. **Build the Toeplitz/BTTB Gram generator.**
   Construct the empirical characteristic function
   [
   T(\delta)
   =========

   \sum_{a,s}
   \frac{\Delta_a}{S}
   \exp(2\pi i, (x_a^{(s)})^\top\delta)
   ]
   on the frequency-difference grid (\delta=\xi_\ell-\xi_k), using a weighted type-1 NUFFT. This defines the BTTB operator
   [
   u\mapsto F_X^*WF_Xu.
   ]

4. **Define the GP posterior precision matvec.**
   For each output dimension (r),
   [
   A_r v
   =====

   v
   +
   \frac{1}{\sigma_r^2}
   D_\theta
   \left[
   F_X^*WF_X
   \right]
   D_\theta v.
   ]
   If (\sigma_r^2) is shared across (r), this is the same operator for every drift dimension.

5. **Build the RHS for the GP posterior mean.**
   First compute
   [
   h_{1,r}
   =======

   D_\theta F_X^*a_r,
   \qquad
   (a_r)_{a,s}
   ===========

   \frac{d_{a,r}}{S\sigma_r^2}.
   ]
   Then compute the derivative correction
   [
   h_{2,r}
   =======

   \sum_{j=1}^{D_{\text{lat}}}
   (2\pi i,\xi_j)
   \odot
   D_\theta F_X^*c_{j,r},
   \qquad
   (c_{j,r})_{a,s}
   ===============

   \frac{C_{a,j,r}}{S\sigma_r^2}.
   ]
   Set
   [
   h_r=h_{1,r}+h_{2,r}.
   ]

6. **Solve for the drift posterior mean.**
   Use CG:
   [
   A_r\mu_r=h_r.
   ]
   Store
   [
   q(w_r)=\mathcal N(\mu_r,A_r^{-1}),
   ]
   but store (A_r^{-1}) only as a solve/apply primitive, not as a dense matrix.

7. **Evaluate EFGP drift moments for the SING latent update.**
   Draw or reuse MC points (X_{\text{eval}}\sim q(x_a)). Compute
   [
   \bar f_r(X_{\text{eval}})=\Phi_{X_{\text{eval}}}\mu_r
   ]
   by type-2 NUFFT. Compute Jacobians by type-2 NUFFTs with coefficients multiplied by (2\pi i\xi_j). Estimate
   [
   \operatorname{diag}(\Phi_{X_{\text{eval}}}A_r^{-1}\Phi_{X_{\text{eval}}}^*)
   ]
   using Hutchinson probes.

8. **Form the transition expectation for SING.**
   Use
   [
   \mathbb E_{q(f)}[f_r(x)^2]
   ==========================

   \bar f_r(x)^2
   +
   \phi(x)^*A_r^{-1}\phi(x)
   ]
   to build the SING transition objective. For diagonal diffusion,
   [
   \mathbb E_{q(f)}
   [
   f(x)^\top\Sigma^{-1}f(x)
   ]
   =

   \sum_r
   \sigma_r^{-2}
   \left[
   \bar f_r(x)^2+
   \phi(x)^*A_r^{-1}\phi(x)
   \right].
   ]

9. **Run the usual SING natural-gradient update for (q(x)).**
   Update the block-tridiagonal natural parameters of (q(x_{0:T})) using the SING update. The only change is that the drift primitives `f`, `ff`, and `dfdx` are now EFGP batched primitives rather than sparse-GP inducing-point expectations.

10. **Repeat inner E-step iterations if desired.**
    Alternate steps 1–9 for several E-step iterations, holding (\theta), emissions, and diffusion fixed. Save M-step updates for later.

---

## 8. Scaling summary

Let (N) be the number of valid transitions, (M=m^d), (S) the number of marginal samples, (k) the number of CG iterations, and (J) the number of Hutchinson probes.

The (q(f)) update costs approximately

[
O(SN + M\log M)
+
O(kD_{\text{out}}M\log M).
]

The (q(x)) drift-moment evaluation costs

[
O(S_{\text{eval}}N + D_{\text{out}}D_{\text{lat}}M\log M)
+
O(J[kM\log M + S_{\text{eval}}N]).
]

So the structure is the desired one:

[
\boxed{
O(N) + f(M),
\qquad
f(M)\approx \text{CG/FFT work } O(kM\log M).
}
]

The implementation rule is simple: **never form (\Phi), never form (\Phi^*W\Phi), never form (Q_r), and never loop over (N) points evaluating all (M) features.** Everything must be NUFFT, BTTB matvec, CG, or SING’s existing Gaussian-chain update.

[1]: https://github.com/lindermanlab/sing/blob/main/sing/sde.py "sing/sing/sde.py at main · lindermanlab/sing · GitHub"
