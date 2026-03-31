import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
from torch.distributions import NegativeBinomial

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from vanilla_gp_sampling import sample_bernoulli_gp
from vanilla_gp_sampling import sample_gp_spectral_approx

from pg_classifier import (
    PolyagammaGPClassifier,
    PolyagammaGPNegativeBinomialRegressor,
    _PGBernoulliLikelihood,
    _VariationalState,
    _build_weighted_toeplitz,
    _build_spectral_state,
    _compute_mstep_gradient,
    _dense_pg_reference_gradient,
    _expected_log_sigmoid_negative_gaussian,
    _make_kernel,
    _make_feature_space_solver,
    _make_sigma_apply,
    _negative_binomial_total_count_gradient,
    _pg_omega_expectation,
    _predictive_variance,
    _predictive_variance_chebyshev,
    _predictive_variance_stochastic,
    _run_estep,
    _solve_beta_mean,
    approximate_logistic_gaussian_prob,
    negative_binomial_gaussian_mean,
)


def test_approximate_logistic_gaussian_prob_matches_notebook_formula():
    mean = torch.tensor([0.2, -0.5, 1.1], dtype=torch.float64)
    variance = torch.tensor([0.1, 0.3, 0.0], dtype=torch.float64)
    expected = torch.sigmoid(mean / torch.sqrt(1.0 + (torch.pi / 8.0) * variance))
    got = approximate_logistic_gaussian_prob(mean, variance)
    assert torch.allclose(got, expected)


def test_pg_omega_expectation_matches_bernoulli_and_nb_formulas():
    c = torch.tensor([0.2, 0.7, 1.3], dtype=torch.float64)
    bern_b = torch.ones_like(c)
    nb_b = torch.tensor([2.5, 4.0, 7.5], dtype=torch.float64)

    bern_expected = torch.tanh(0.5 * c) / (2.0 * c)
    nb_expected = 0.5 * nb_b * torch.tanh(0.5 * c) / c

    assert torch.allclose(_pg_omega_expectation(c, bern_b), bern_expected)
    assert torch.allclose(_pg_omega_expectation(c, nb_b), nb_expected)


def test_negative_binomial_total_count_gradient_matches_finite_difference():
    y = torch.tensor([0.0, 2.0, 5.0, 7.0], dtype=torch.float64)
    mean = torch.tensor([-0.4, 0.2, 0.9, 1.3], dtype=torch.float64)
    variance = torch.tensor([0.1, 0.5, 0.2, 0.7], dtype=torch.float64)
    total_count = 2.4

    grad = _negative_binomial_total_count_gradient(
        y,
        mean,
        variance,
        total_count=total_count,
        quadrature_nodes=16,
    )

    def objective(r: float) -> float:
        expected_log_sigmoid_neg = _expected_log_sigmoid_negative_gaussian(
            mean,
            variance,
            quadrature_nodes=64,
        )
        r_t = torch.tensor(r, dtype=torch.float64)
        value = torch.sum(
            torch.lgamma(y + r_t)
            - torch.lgamma(r_t)
            + r_t * expected_log_sigmoid_neg
        )
        return float(value.item())

    eps = 1e-5
    reference = (objective(total_count + eps) - objective(total_count - eps)) / (2.0 * eps)
    assert math.isfinite(reference)
    assert abs(grad.item() - reference) < 5e-4


def test_polyagamma_gp_classifier_sklearn_api():
    torch.manual_seed(0)
    X = torch.rand(24, 1, dtype=torch.float64) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=1.0)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy().astype(int)
    X_test = np.linspace(-1.25, 1.25, 9, dtype=np.float64).reshape(-1, 1)

    clf = PolyagammaGPClassifier(
        lengthscale_init=0.3,
        variance_init=1.0,
        max_iter=2,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=4,
        n_m_probes=8,
        random_state=123,
        store_history=True,
    )
    clf.fit(X_np, y_np)

    logits = clf.decision_function(X_np)
    proba = clf.predict_proba(X_np)
    pred = clf.predict(X_np)
    test_logits = clf.decision_function(X_test)
    test_var = clf.predictive_variance(X_test)
    test_proba = clf.predict_proba(X_test)

    assert logits.shape == (X_np.shape[0],)
    assert proba.shape == (X_np.shape[0], 2)
    assert pred.shape == (X_np.shape[0],)
    assert test_logits.shape == (X_test.shape[0],)
    assert test_var.shape == (X_test.shape[0],)
    assert test_proba.shape == (X_test.shape[0], 2)
    assert np.all(np.isfinite(logits))
    assert np.all(np.isfinite(proba))
    assert np.all(np.isfinite(test_logits))
    assert np.all(np.isfinite(test_var))
    assert np.all(np.isfinite(test_proba))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(test_proba.sum(axis=1), 1.0)
    assert set(np.unique(pred)).issubset(set(clf.classes_))
    assert clf.posterior_mean_.shape == (X_np.shape[0],)
    assert clf.posterior_var_diag_.shape == (X_np.shape[0],)
    assert clf.delta_.shape == (X_np.shape[0],)
    assert np.all(clf.delta_ >= 0.0)
    assert np.all(test_var >= 0.0)
    assert 0.0 <= clf.training_accuracy_ <= 1.0
    assert len(clf.history_) == clf.max_iter + 1

    approx_test = approximate_logistic_gaussian_prob(
        torch.as_tensor(test_logits, dtype=torch.float64),
        torch.as_tensor(test_var, dtype=torch.float64),
    ).cpu().numpy()
    assert np.allclose(test_proba[:, 1], approx_test)


def test_feature_space_mstep_matches_dense_pg_reference_small_problem():
    torch.manual_seed(7)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(14, 1, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.35, variance=1.0)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.3,
        variance=1.0,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )
    mstep = _compute_mstep_gradient(
        kappa,
        variational.delta,
        spectral,
        n_probes=256,
        cg_tol=1e-7,
        seed=0,
    )
    reference = _dense_pg_reference_gradient(
        X,
        y,
        variational.mean,
        variational.delta,
        spectral,
        jitter=1e-7,
    )

    assert torch.sign(mstep["grad"][0]) == torch.sign(reference[0])
    assert torch.sign(mstep["grad"][1]) == torch.sign(reference[1])
    assert abs((mstep["grad"][0] - reference[0]).item()) / abs(reference[0].item()) < 0.6
    assert abs((mstep["grad"][1] - reference[1]).item()) < 0.15


def test_weighted_toeplitz_estep_matches_nufft_path_small_problem():
    torch.manual_seed(11)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(16, 1, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.35, variance=1.0)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.3,
        variance=1.0,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    base_delta = torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device)

    nufft_state, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        _VariationalState(delta=base_delta.clone()),
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-8,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        use_exact_weighted_toeplitz_operator=False,
        seed=123,
        verbose=0,
    )
    toeplitz_state, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        _VariationalState(delta=base_delta.clone()),
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-8,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        use_exact_weighted_toeplitz_operator=True,
        seed=123,
        verbose=0,
    )

    assert torch.allclose(nufft_state.mean, toeplitz_state.mean, atol=5e-6, rtol=5e-6)
    assert torch.allclose(nufft_state.sigma_diag, toeplitz_state.sigma_diag, atol=5e-6, rtol=5e-6)
    assert torch.allclose(nufft_state.delta, toeplitz_state.delta, atol=5e-6, rtol=5e-6)


def test_weighted_toeplitz_beta_mean_matches_nufft_path_small_problem():
    torch.manual_seed(13)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(14, 1, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=1.0)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.35,
        variance=1.0,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )

    beta_nufft, *_ = _solve_beta_mean(
        kappa,
        variational.delta,
        spectral,
        cg_tol=1e-7,
        use_exact_weighted_toeplitz_operator=False,
    )
    beta_toeplitz, *_ = _solve_beta_mean(
        kappa,
        variational.delta,
        spectral,
        cg_tol=1e-7,
        use_exact_weighted_toeplitz_operator=True,
    )

    assert torch.allclose(beta_nufft, beta_toeplitz, atol=5e-6, rtol=5e-6)


def test_weighted_toeplitz_training_solve_matches_dense_feature_space():
    torch.manual_seed(123)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(18, 1, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.35, variance=1.0)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.3,
        variance=1.0,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        use_exact_weighted_toeplitz_operator=False,
        seed=0,
        verbose=0,
    )

    delta = variational.delta
    omega = delta.to(dtype=cdtype)
    Ds = torch.sqrt(torch.clamp(spectral.ws2.real, min=1e-14)).to(dtype=cdtype)
    F = torch.exp(2j * torch.pi * (X @ spectral.xis.T)).to(dtype=cdtype)
    A_dense = torch.eye(F.shape[1], dtype=cdtype, device=device) + (
        Ds[:, None] * (F.conj().T @ (omega[:, None] * F)) * Ds[None, :]
    )
    A_dense = 0.5 * (A_dense + A_dense.conj().T)

    v = torch.randn(F.shape[1], dtype=cdtype, device=device)
    from pg_classifier import _build_weighted_toeplitz
    T_weighted = _build_weighted_toeplitz(delta.to(dtype=cdtype), spectral)
    A_toeplitz_v = v + Ds * T_weighted(Ds * v)
    act_rel = (torch.linalg.norm(A_toeplitz_v - A_dense @ v) / torch.linalg.norm(A_dense @ v)).item()

    solve_A_beta, _, _ = _make_feature_space_solver(
        delta,
        spectral,
        cg_tol=1e-10,
        use_exact_weighted_toeplitz_operator=True,
    )
    q = spectral.fadj(kappa.to(dtype=cdtype))
    rhs = Ds * q
    beta_dense = torch.linalg.solve(A_dense, rhs[:, None]).squeeze(1) / Ds
    beta_toeplitz, _ = solve_A_beta(q)
    beta_rel = (torch.linalg.norm(beta_toeplitz - beta_dense) / torch.linalg.norm(beta_dense)).item()

    sigma_apply, _ = _make_sigma_apply(
        spectral,
        delta,
        cg_tol=1e-10,
        use_exact_weighted_toeplitz_operator=True,
    )
    Z = torch.randn(4, X.shape[0], dtype=rdtype, device=device)
    sigma_toeplitz = sigma_apply(Z)
    Psi = F * spectral.ws[None, :]
    K = Psi @ Psi.conj().T
    Sigma_dense = torch.linalg.solve(
        torch.eye(X.shape[0], dtype=cdtype, device=device) + K * omega[None, :],
        K,
    )
    sigma_dense = (Sigma_dense @ Z.to(dtype=cdtype).T).T.real
    sigma_rel = (torch.linalg.norm(sigma_toeplitz - sigma_dense) / torch.linalg.norm(sigma_dense)).item()

    assert act_rel < 1e-7
    assert beta_rel < 1e-7
    assert sigma_rel < 1e-7


def test_weighted_toeplitz_predictive_variance_matches_nufft_exact_path():
    torch.manual_seed(17)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(12, 1, dtype=rdtype, device=device) * 2 - 1
    X_test = torch.linspace(-0.95, 0.95, 7, dtype=rdtype, device=device).unsqueeze(-1)
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=1.0)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.35,
        variance=1.0,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-4,
        trunc_eps=1e-4,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-7,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )

    var_nufft = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-7,
        nufft_eps=1e-7,
        use_exact_weighted_toeplitz_operator=False,
        batch_size=4,
    )
    var_toep = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-7,
        nufft_eps=1e-7,
        use_exact_weighted_toeplitz_operator=True,
        batch_size=4,
    )

    assert torch.allclose(var_nufft, var_toep, atol=5e-6, rtol=5e-6)


def test_predictive_variance_matches_dense_feature_space_reference():
    torch.manual_seed(11)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(10, 1, dtype=rdtype, device=device) * 2 - 1
    X_test = torch.linspace(-0.9, 0.9, 4, dtype=rdtype, device=device).unsqueeze(-1)
    y, _ = sample_bernoulli_gp(X, length_scale=0.45, variance=0.8)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.35,
        variance=0.9,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-3,
        trunc_eps=1e-3,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-8,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )
    variance = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        batch_size=2,
    )

    train_basis = torch.eye(X.shape[0], device=device, dtype=cdtype)
    test_basis = torch.eye(X_test.shape[0], device=device, dtype=cdtype)
    train_features = spectral.nufft_op.type1(train_basis, out_shape=spectral.out_shape).reshape(X.shape[0], -1)
    test_op = spectral.nufft_op.__class__(
        X_test,
        torch.zeros_like(X_test),
        spectral.h,
        1e-7,
        cdtype=cdtype,
        device=device,
    )
    test_features = test_op.type1(test_basis, out_shape=spectral.out_shape).reshape(X_test.shape[0], -1)

    K_train = spectral.fwd_batched(spectral.ws2.unsqueeze(0) * train_features).real
    K_cross = spectral.fwd_batched(spectral.ws2.unsqueeze(0) * test_features).real
    k_ss = test_op.type2(spectral.ws2.unsqueeze(0) * test_features, out_shape=spectral.out_shape).real.diagonal()

    A = torch.eye(X.shape[0], device=device, dtype=rdtype) + variational.delta.unsqueeze(1) * K_train
    rhs = (variational.delta.unsqueeze(1) * K_cross.T)
    alpha = torch.linalg.solve(A, rhs)
    reference = k_ss - torch.sum(K_cross.T * alpha, dim=0)

    assert torch.all(variance >= 0.0)
    assert torch.allclose(variance, reference, atol=1e-5, rtol=1e-4)


def test_stochastic_predictive_variance_tracks_exact_small_problem():
    torch.manual_seed(19)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(12, 1, dtype=rdtype, device=device) * 2 - 1
    X_test = torch.linspace(-1.0, 1.0, 6, dtype=rdtype, device=device).unsqueeze(-1)
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=0.9)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=0.35,
        variance=0.9,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-3,
        trunc_eps=1e-3,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-8,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )

    exact = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        batch_size=3,
    )
    stochastic, info = _predictive_variance_stochastic(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        n_probes=256,
        seed=123,
    )

    assert info["n_probes"] == 256
    assert torch.all(torch.isfinite(stochastic))
    assert torch.all(stochastic >= 0.0)
    assert torch.max(torch.abs(stochastic - exact)).item() < 0.12
    assert torch.mean(torch.abs(stochastic - exact)).item() < 0.05


def test_chebyshev_predictive_variance_tracks_exact_small_problem():
    torch.manual_seed(31)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(12, 2, dtype=rdtype, device=device) * 2 - 1
    X_test = torch.rand(18, 2, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=0.9)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=2,
        lengthscale=0.35,
        variance=0.9,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-3,
        trunc_eps=1e-3,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-8,
        reuse_probes=True,
        seed=0,
        verbose=0,
    )

    exact = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        batch_size=6,
    )
    cheb, info = _predictive_variance_chebyshev(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        n_nodes_per_dim=7,
        batch_size=6,
    )

    assert info["n_nodes_total"] == 49.0
    assert torch.all(torch.isfinite(cheb))
    assert torch.all(cheb >= 0.0)
    assert torch.max(torch.abs(cheb - exact)).item() < 0.08
    assert torch.mean(torch.abs(cheb - exact)).item() < 0.03


def test_cached_weighted_toeplitz_matches_uncached_predictive_variance_paths():
    torch.manual_seed(41)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(14, 2, dtype=rdtype, device=device) * 2 - 1
    X_test = torch.rand(19, 2, dtype=rdtype, device=device) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=0.9)
    y = y.to(device=device, dtype=rdtype)

    kernel = _make_kernel(
        "squared_exponential",
        dimension=2,
        lengthscale=0.35,
        variance=0.9,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=1e-3,
        trunc_eps=1e-3,
        nufft_eps=1e-7,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    likelihood = _PGBernoulliLikelihood()
    kappa = likelihood.kappa(y)
    pg_b = likelihood.pg_b(y)
    variational = _VariationalState(delta=torch.full((X.shape[0],), 0.25, dtype=rdtype, device=device))
    variational, _ = _run_estep(
        y,
        kappa,
        pg_b,
        likelihood,
        variational,
        spectral,
        max_iters=1,
        rho0=0.7,
        gamma=1e-3,
        tol=1e-6,
        n_probes=8,
        cg_tol=1e-8,
        reuse_probes=True,
        use_exact_weighted_toeplitz_operator=True,
        seed=0,
        verbose=0,
    )

    weighted_toeplitz = _build_weighted_toeplitz(
        variational.delta.to(dtype=spectral.ws.dtype),
        spectral,
    )

    exact_uncached = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        use_exact_weighted_toeplitz_operator=True,
        batch_size=6,
    )
    exact_cached = _predictive_variance(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        use_exact_weighted_toeplitz_operator=True,
        batch_size=6,
        weighted_toeplitz=weighted_toeplitz,
    )
    cheb_uncached, _ = _predictive_variance_chebyshev(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        n_nodes_per_dim=7,
        use_exact_weighted_toeplitz_operator=True,
        batch_size=6,
    )
    cheb_cached, _ = _predictive_variance_chebyshev(
        X_test,
        variational.delta,
        spectral,
        cg_tol=1e-8,
        nufft_eps=1e-7,
        n_nodes_per_dim=7,
        use_exact_weighted_toeplitz_operator=True,
        batch_size=6,
        weighted_toeplitz=weighted_toeplitz,
    )

    assert torch.allclose(exact_uncached, exact_cached, atol=1e-10, rtol=1e-10)
    assert torch.allclose(cheb_uncached, cheb_cached, atol=1e-10, rtol=1e-10)


def test_spectral_gp_sampler_matches_approximate_kernel_covariance():
    torch.manual_seed(23)
    device = torch.device("cpu")
    rdtype = torch.float64
    cdtype = torch.complex128

    X = torch.rand(6, 1, dtype=rdtype, device=device) * 2 - 1
    n_samples = 512
    length_scale = 0.45
    variance = 0.9
    spectral_eps = 1e-4
    trunc_eps = 1e-4
    nufft_eps = 1e-7

    samples = sample_gp_spectral_approx(
        X,
        num_samples=n_samples,
        length_scale=length_scale,
        variance=variance,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        seed=123,
    )

    kernel = _make_kernel(
        "squared_exponential",
        dimension=1,
        lengthscale=length_scale,
        variance=variance,
    )
    spectral = _build_spectral_state(
        X,
        kernel,
        spectral_eps=spectral_eps,
        trunc_eps=trunc_eps,
        nufft_eps=nufft_eps,
        rdtype=rdtype,
        cdtype=cdtype,
        device=device,
    )
    F_train = torch.exp(2.0 * math.pi * 1j * (X @ spectral.xis.T)).to(dtype=cdtype)
    K_approx = (F_train @ torch.diag(spectral.ws2.to(dtype=cdtype)) @ F_train.T.conj()).real

    centered = samples - samples.mean(dim=1, keepdim=True)
    empirical = centered @ centered.T / samples.shape[1]

    diag_rel = torch.linalg.norm(torch.diag(empirical) - torch.diag(K_approx)) / torch.linalg.norm(torch.diag(K_approx))
    fro_rel = torch.linalg.norm(empirical - K_approx) / torch.linalg.norm(K_approx)

    assert samples.shape == (X.shape[0], n_samples)
    assert torch.all(torch.isfinite(samples))
    assert fro_rel.item() < 0.35
    assert diag_rel.item() < 0.20


def test_negative_binomial_regressor_fixed_total_count_api():
    torch.manual_seed(5)
    X = torch.rand(32, 1, dtype=torch.float64) * 2 - 1
    latent = 0.8 * torch.sin(2.5 * X.squeeze(-1))
    total_count = 3.0
    y = NegativeBinomial(
        total_count=torch.tensor(total_count, dtype=torch.float64),
        logits=latent,
    ).sample()

    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    X_test = np.linspace(-1.25, 1.25, 11, dtype=np.float64).reshape(-1, 1)

    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=total_count,
        lengthscale_init=0.35,
        variance_init=1.0,
        max_iter=2,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=4,
        n_m_probes=8,
        random_state=321,
        store_history=True,
    )
    reg.fit(X_np, y_np)

    latent_train = reg.decision_function(X_np)
    latent_var_train = reg.predictive_variance(X_np)
    mean_count_train = reg.predict(X_np)
    mean_count_test = reg.predict_mean_count(X_test)
    latent_test = reg.decision_function(X_test)
    latent_var_test = reg.predictive_variance(X_test)

    assert latent_train.shape == (X_np.shape[0],)
    assert latent_var_train.shape == (X_np.shape[0],)
    assert mean_count_train.shape == (X_np.shape[0],)
    assert mean_count_test.shape == (X_test.shape[0],)
    assert latent_test.shape == (X_test.shape[0],)
    assert latent_var_test.shape == (X_test.shape[0],)
    assert np.all(np.isfinite(mean_count_train))
    assert np.all(np.isfinite(mean_count_test))
    assert np.all(mean_count_train >= 0.0)
    assert np.all(mean_count_test >= 0.0)
    assert np.all(reg.delta_ >= 0.0)
    assert len(reg.history_) == reg.max_iter + 1
    assert np.isfinite(reg.training_mean_absolute_error_)
    assert math.isclose(reg.total_count_, total_count)

    expected_train = negative_binomial_gaussian_mean(
        torch.as_tensor(latent_train, dtype=torch.float64),
        torch.as_tensor(latent_var_train, dtype=torch.float64),
        total_count=total_count,
    ).cpu().numpy()
    assert np.allclose(mean_count_train, expected_train)


def test_classifier_stochastic_predictive_variance_api_is_reproducible():
    torch.manual_seed(29)
    X = torch.rand(18, 1, dtype=torch.float64) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=1.0)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy().astype(int)
    X_test = np.linspace(-1.0, 1.0, 13, dtype=np.float64).reshape(-1, 1)

    clf = PolyagammaGPClassifier(
        lengthscale_init=0.35,
        variance_init=1.0,
        max_iter=2,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=4,
        n_m_probes=6,
        predictive_variance_method="stochastic",
        predictive_variance_probes=64,
        random_state=777,
    )
    clf.fit(X_np, y_np)
    var1 = clf.predictive_variance(X_test)
    var2 = clf.predictive_variance(X_test)
    proba = clf.predict_proba(X_test)

    assert np.all(np.isfinite(var1))
    assert np.all(var1 >= 0.0)
    assert np.allclose(var1, var2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert clf._stochastic_predictive_variance_info_["n_probes"] == 64


def test_classifier_chebyshev_predictive_variance_api_is_reproducible():
    torch.manual_seed(37)
    X = torch.rand(18, 2, dtype=torch.float64) * 2 - 1
    y, _ = sample_bernoulli_gp(X, length_scale=0.4, variance=1.0)
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy().astype(int)
    X_test = np.random.default_rng(0).uniform(-1.1, 1.1, size=(21, 2))

    clf = PolyagammaGPClassifier(
        lengthscale_init=0.35,
        variance_init=1.0,
        max_iter=2,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=4,
        n_m_probes=6,
        predictive_variance_method="chebyshev",
        predictive_variance_chebyshev_nodes=7,
        random_state=777,
    )
    clf.fit(X_np, y_np)
    var1 = clf.predictive_variance(X_test)
    var2 = clf.predictive_variance(X_test)
    proba = clf.predict_proba(X_test)

    assert np.all(np.isfinite(var1))
    assert np.all(var1 >= 0.0)
    assert np.allclose(var1, var2)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_negative_binomial_regressor_can_learn_total_count():
    torch.manual_seed(19)
    X = torch.rand(48, 1, dtype=torch.float64) * 2 - 1
    latent = 0.25 * torch.sin(2.0 * math.pi * X.squeeze(-1))
    true_total_count = 4.0
    init_total_count = 1.5
    y = NegativeBinomial(
        total_count=torch.tensor(true_total_count, dtype=torch.float64),
        logits=latent,
    ).sample()

    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    reg = PolyagammaGPNegativeBinomialRegressor(
        total_count=init_total_count,
        learn_total_count=True,
        total_count_lr=0.08,
        total_count_update_frequency=1,
        total_count_quadrature_nodes=16,
        lengthscale_init=0.35,
        variance_init=0.8,
        max_iter=6,
        e_step_iters=1,
        final_e_step_iters=1,
        n_e_probes=4,
        n_m_probes=8,
        random_state=123,
        store_history=True,
    )
    reg.fit(X_np, y_np)

    pred = reg.predict(X_np)
    assert pred.shape == (X_np.shape[0],)
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0)
    assert np.isfinite(reg.total_count_)
    assert reg.total_count_ > 0.0
    assert abs(reg.total_count_ - true_total_count) < abs(init_total_count - true_total_count)
    assert len(reg.history_) == reg.max_iter + 1

    total_count_path = np.array([row["total_count"] for row in reg.history_], dtype=float)
    grad_path = np.array([row["grad_total_count"] for row in reg.history_], dtype=float)
    updated = np.array([row["total_count_updated"] for row in reg.history_[:-1]], dtype=float)

    assert np.all(np.isfinite(total_count_path))
    assert np.all(total_count_path > 0.0)
    assert np.all(np.isfinite(grad_path))
    assert np.any(np.abs(total_count_path[:-1] - init_total_count) > 1e-3)
    assert np.all(updated == 1.0)
