# What Is Actually Bad About the $\sigma_n^2$ Gradient?

This note records the specific computation that becomes pathological in the
small-lengthscale / large-variance / small-noise regime for `efgpnd.py`.

The short version is:

- The bad computation is not the whole hypergradient uniformly.
- It is overwhelmingly the stochastic trace term for the noise variance.
- More specifically, it is the CG solve for the noise-block probe systems inside
  the batched Hutchinson trace estimator.

## Gradient Split

The log marginal likelihood gradient has the usual form

$\nabla_\theta \log p(y \mid \theta)
= \frac{1}{2}
\left[
\operatorname{tr}\!\left(K^{-1} \frac{\partial K}{\partial \theta}\right)
- y^\top K^{-1}
\left(\frac{\partial K}{\partial \theta}\right)
K^{-1} y
\right].$

In `efgpnd.py`, these are the two pieces:

- `term1`: the stochastic Hutchinson trace term
- `term2`: the quadratic form

If we define

$$
\alpha = K^{-1} y,
$$

then the second term can be written as

$$
y^\top K^{-1}
\left(\frac{\partial K}{\partial \theta}\right)
K^{-1} y
=
\alpha^\top
\left(\frac{\partial K}{\partial \theta}\right)
\alpha.
$$

For the noise variance parameter,

$$
\frac{\partial K}{\partial \sigma_n^2} = I,
$$

so the gradient pieces specialize to

$$
\operatorname{term2}_{\sigma_n^2} = \alpha^\top \alpha,
$$

and

$$
\operatorname{term1}_{\sigma_n^2}
=
\operatorname{tr}(K^{-1}).
$$

The key point is that $\alpha^\top \alpha$ is cheap once $\alpha$ has already
been computed. The hard object is $\operatorname{tr}(K^{-1})$.

## The Specific Bad Computation

Let the approximate kernel be written in feature form as

$$
\widetilde K = \Phi \Phi^\ast,
\qquad
\Phi = F D,
$$

where:

- $F$ is the nonuniform Fourier operator used in the code
- $D$ is the diagonal matrix of spectral weights

Then the data-space system is

$$
K = \widetilde K + \sigma_n^2 I
=
\Phi \Phi^\ast + \sigma_n^2 I.
$$

The CG system that the implementation actually solves is written in feature
space:

$$
A \beta = b,
\qquad
A = D(F^\ast F)D + \sigma_n^2 I.
$$

For the noise-variance trace block, the right-hand side is built from white
noise probe vectors $z$, and the core computation is

$$
A^{-1}(D F^\ast z).
$$

That is the slow part.

## Why the Noise Block Is Much Worse Than the Other Hyperparameters

For the kernel hyperparameters, the trace estimator probes

$$
\frac{\partial K}{\partial \theta}
$$

through spectrally weighted directions. In code, those right-hand sides are
multiplied by derivative factors such as $D'$, and then filtered through the
Toeplitz operator again. Those probes are structured and relatively smooth.

For the noise parameter, there is no such filtering:

$$
\frac{\partial K}{\partial \sigma_n^2} = I.
$$

So the trace term asks for

$$
\operatorname{tr}(K^{-1}),
$$

which means the Hutchinson estimator probes all modes of $K^{-1}$, including
the weakly regularized ones. When $\sigma_n^2$ is small, those are exactly the
directions that are hardest for CG.

That is why the noise block is much worse than the lengthscale or variance
blocks even when they are solved inside the same batched CG call.

## Real-Data Evidence

I used standardized `usa_temp_data` with the standalone diagnostics:

- `benchmark_cg_preconditioning_realdata.py`
- `diagnose_sigma_noise_trace_blocks.py`

In a hard regime with

$$
\ell = 0.03,
\qquad
\sigma_f^2 = 10,
\qquad
\varepsilon = 10^{-4},
\qquad
J = 3,
\qquad
\text{cg\_tol} = 10^{-3},
\qquad
n = 4766,
$$

the per-RHS CG iteration counts were:

### When $\sigma_n^2 = 10^{-4}$

- $d\ell$: $59, 59, 47$
- $d\sigma_f^2$: $29, 30, 24$
- $d\sigma_n^2$: $4162, 3945, 3667$

### When $\sigma_n^2 = 10^{-3}$

- $d\ell$: $59, 59, 47$
- $d\sigma_f^2$: $29, 30, 24$
- $d\sigma_n^2$: $1643, 1631, 1588$

### When $\sigma_n^2 = 10^{-2}$

- $d\ell$: $59, 59, 47$
- $d\sigma_f^2$: $29, 29, 24$
- $d\sigma_n^2$: $634, 629, 606$

So increasing $\sigma_n^2$ helps a lot, but even at $\sigma_n^2 = 10^{-2}$ the
noise block is still an order of magnitude harder than the kernel-hyper blocks.
That means tiny noise is not the whole story. The deeper issue is that the
$\sigma_n^2$ trace term is structurally the wrong kind of probe for this solver.

## Why a Floor Helps but Does Not Fix the Problem

In the same hard real-data regime, a simple diagonal proxy for the feature-space
operator ranged roughly from

$$
10^{-4}
\quad \text{to} \quad
2 \times 10^2,
$$

with a median around

$$
4 \times 10^{-2}.
$$

That is a dynamic range on the order of

$$
2 \times 10^6.
$$

When a floor or jitter is added, the weakest directions get more diagonal
regularization, so the worst iteration counts come down. But the estimator is
still trying to approximate

$$
\operatorname{tr}(K^{-1}),
$$

which means the same structurally difficult quantity is still present. The floor
reduces the severity of the problem, but it does not remove the underlying cause.

## Clamp Versus Additive Jitter

Two common numerical modifications are

$$
\sigma_{\mathrm{eff}}^2
=
\max(\sigma_n^2, \tau^2)
$$

and

$$
\sigma_{\mathrm{eff}}^2
=
\sigma_n^2 + \tau^2.
$$

Both change the effective operator being solved.

The additive version is smoother in $\sigma_n^2$ and easier to interpret as
"learned noise plus fixed numerical diagonal regularization." But it is still a
numerical or modeling intervention, not an exact optimization of the original
objective.

So additive jitter is more coherent than a hard clamp, but it is not magically
free of bias.

## A More Defensible Way to Treat the Noise Gradient

The publishable fix is not "do not learn $\sigma_n^2$." The better story is:

- keep the current stochastic machinery for the kernel hyperparameters
- treat $\sigma_n^2$ separately, because its derivative is algebraically special

Start from

$$
K = \sigma_n^2 I_n + \Phi \Phi^\ast,
$$

with $\Phi \in \mathbb{C}^{n \times M}$ and

$$
S = \Phi^\ast \Phi = D(F^\ast F)D.
$$

Then the matrix determinant lemma gives

$$
\det(\sigma_n^2 I_n + \Phi \Phi^\ast)
=
(\sigma_n^2)^{n-M} \det(\sigma_n^2 I_M + S),
$$

or equivalently,

$$
\log \det(\sigma_n^2 I_n + \Phi \Phi^\ast)
=
n \log \sigma_n^2
+
\log \det\!\left(I_M + \sigma_n^{-2} S\right).
$$

Likewise, the nonzero eigenvalues of $\Phi \Phi^\ast$ and $S = \Phi^\ast \Phi$
match, so

$$
\operatorname{tr}\!\left((\sigma_n^2 I_n + \Phi \Phi^\ast)^{-1}\right)
=
\frac{n-M}{\sigma_n^2}
+
\operatorname{tr}\!\left((\sigma_n^2 I_M + S)^{-1}\right),
$$

which can also be written as

$$
\operatorname{tr}\!\left((\sigma_n^2 I_n + \Phi \Phi^\ast)^{-1}\right)
=
\frac{n-M}{\sigma_n^2}
+
\frac{1}{\sigma_n^2}
\operatorname{tr}\!\left(\left(I_M + \sigma_n^{-2} S\right)^{-1}\right).
$$

This is the crucial identity. It says that the noise-variance trace term can be
reduced to a feature-space trace, instead of a data-space white-noise Hutchinson
problem.

If we write the $\sigma_n^2$ gradient as

$$
\frac{\partial}{\partial \sigma_n^2} \log p(y \mid \theta)
=
\frac{1}{2}
\left[
\operatorname{tr}(K^{-1}) - \alpha^\top \alpha
\right],
$$

then the trace piece becomes

$$
\operatorname{tr}(K^{-1})
=
\frac{n-M}{\sigma_n^2}
+
\frac{1}{\sigma_n^2}
\operatorname{tr}\!\left(\left(I_M + \sigma_n^{-2} S\right)^{-1}\right).
$$

So the entire noise gradient can be written as

$$
\frac{\partial}{\partial \sigma_n^2} \log p(y \mid \theta)
=
\frac{1}{2}
\left[
\frac{n-M}{\sigma_n^2}
+
\frac{1}{\sigma_n^2}
\operatorname{tr}\!\left(\left(I_M + \sigma_n^{-2} S\right)^{-1}\right)
- \alpha^\top \alpha
\right].
$$

This is much more defensible than clamping alone, because it uses the algebraic
structure of the noise hyperparameter instead of forcing the same noisy estimator
used for the kernel hyperparameters onto a fundamentally different derivative.

## Bottom Line

The specific bad computation is

$$
\operatorname{tr}(K^{-1}),
$$

estimated stochastically through the $\sigma_n^2$ noise block, which requires
repeated solves of the form

$$
\left(D(F^\ast F)D + \sigma_n^2 I\right)^{-1}(D F^\ast z).
$$

That is the pathological part of the current method.

The quadratic form

$$
\alpha^\top \alpha
$$

is not the problem.

If $\sigma_n^2 = 10^{-2}$ is still not acceptable, then the issue is not just
"noise too small." It means the stochastic trace formulation for the
$\sigma_n^2$ hypergradient is intrinsically the hard piece of the current method.
