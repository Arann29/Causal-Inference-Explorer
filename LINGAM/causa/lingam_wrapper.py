import numpy as np
import lingam as lingam_pkg


def lingam_wrapper(x, y, device="cpu", random_state=711):
    """
    Run bivariate DirectLiNGAM.

    Returns a signed score:
        score > 0: X -> Y
        score < 0: Y -> X
        score = 0: no clear direction

    Notes:
        The `device` argument is kept only for compatibility with the common
        wrapper interface. DirectLiNGAM does not use GPU/CPU device settings.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 3:
        raise ValueError("LiNGAM requires at least 3 samples")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("x and y must not contain NaN values")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("x and y must not contain Inf values")

    data = np.column_stack([x, y])

    model = lingam_pkg.DirectLiNGAM(random_state=random_state)
    model.fit(data)

    adjacency = np.asarray(model.adjacency_matrix_, dtype=float)

    if adjacency.shape != (2, 2):
        return 0.0

    # In lingam, adjacency[i, j] means j -> i.
    x_to_y = abs(adjacency[1, 0])
    y_to_x = abs(adjacency[0, 1])

    score = x_to_y - y_to_x

    if np.isclose(score, 0.0):
        order = list(model.causal_order_)

        if order == [0, 1]:
            return 1e-6
        if order == [1, 0]:
            return -1e-6

        return 0.0

    return float(score)