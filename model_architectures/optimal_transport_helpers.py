import torch


def batched_optimal_transport_log(
    dist_matrix: torch.Tensor,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    reg: float = 0.1,
    threshold: float = 0.01,
    max_iter: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the optimal transport between the items in the dist_matrix for a batch of images in the log space

    :param dist_matrix: [N, P, H1, H2] tensor containing the cost for N images (batch dimension) to all P prototypes
                        of all patches of 1 images (H1) versus the patches in the other image (H2)
    :param weights_a: prior for the H1 dimension of size [N,P,H1] or [H1]
    :param weights_b: prior for the H2 dimension of size [N,P,H2] or [H2]
    :param reg: magnitude of the entropic regularization (temperature), defaults to 0.1
    :param threshold: threshold at which to stop iteration (not really used if any infinites appear in result)
    :param max_iter: maximum iterations to do for the optimal transport computation
    :return: the optimal flow matrix of same size as dist_matrix
    """

    # assert the sizes are correct
    assert weights_a.shape == weights_b.shape
    assert len(weights_a.shape) == 3 or len(weights_a.shape) == 1

    # this is the case of the weights being [N, P, H1]
    if len(weights_a.shape) == 3:
        mu, nu = weights_a, weights_b
    else:
        # assume same weight for all priors (i.e. for uniform weights)
        mu, nu = weights_a.view([1, 1, -1]), weights_b.view([1, 1, -1])

    # initialize u and v
    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    logmu = torch.log(mu)
    lognu = torch.log(nu)

    # convert dist matrix into matrix scaling problem with entropic regularization
    K = -dist_matrix / reg

    # iteratively update u and v using Sinkhorn iterations (i.e. matrix scaling)
    for _ in range(max_iter):
        u1 = u

        u = logmu - torch.logsumexp(K + v[:, :, None, :], dim=3)
        v = lognu - torch.logsumexp(K + u[:, :, :, None], dim=2)

        # check whether update step has been smaller than threshold
        err = (u - u1).abs().max()
        if err < threshold:
            break

    # Compute optimal flow matrix
    T = torch.exp(K + u[:, :, :, None] + v[:, :, None, :])

    return T, err
