import numpy as np

from chebyshev_variance_experiment import (
    barycentric_interpolation_matrix,
    chebyshev_lobatto_nodes,
    tensor_product_chebyshev_interpolate_2d,
)


def test_barycentric_matrix_reproduces_polynomial_in_1d():
    nodes, weights = chebyshev_lobatto_nodes(-1.5, 2.0, 6)
    targets = np.linspace(-1.5, 2.0, 17)
    interp = barycentric_interpolation_matrix(nodes, weights, targets)

    poly = lambda x: 1.0 - 2.0 * x + 0.5 * x**2 - 0.25 * x**3 + 0.1 * x**4
    values = poly(nodes)
    approx = interp @ values
    truth = poly(targets)

    assert np.allclose(approx, truth, atol=1e-10, rtol=1e-10)


def test_tensor_product_chebyshev_interpolation_reproduces_bivariate_polynomial():
    x_nodes, x_weights = chebyshev_lobatto_nodes(-1.0, 1.0, 5)
    y_nodes, y_weights = chebyshev_lobatto_nodes(-2.0, 2.0, 6)
    x_targets = np.linspace(-1.0, 1.0, 13)
    y_targets = np.linspace(-2.0, 2.0, 11)

    interp_x = barycentric_interpolation_matrix(x_nodes, x_weights, x_targets)
    interp_y = barycentric_interpolation_matrix(y_nodes, y_weights, y_targets)

    xx_nodes, yy_nodes = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    values = 1.0 + xx_nodes**2 - 0.5 * xx_nodes * yy_nodes + 0.25 * yy_nodes**3

    approx = tensor_product_chebyshev_interpolate_2d(values, interp_x, interp_y)
    xx_targets, yy_targets = np.meshgrid(x_targets, y_targets, indexing="ij")
    truth = 1.0 + xx_targets**2 - 0.5 * xx_targets * yy_targets + 0.25 * yy_targets**3

    assert np.allclose(approx, truth, atol=1e-10, rtol=1e-10)
