import torch
from torch import nn
from define_parameters import LossCoefs
from model_architectures.ExPeRT_arch import PPNet_wholeprotos
from typing import Optional, Type

ProtoNet = Type[PPNet_wholeprotos]


def get_position_enc_flip(dim1: int, dim2: int) -> torch.Tensor:
    """Generate a matrix of size [25x25]

    :param size: input size [R], defaults to 25
    :return: matrix of [R, R] with 1 at the corresponding locations before and after the horizontal flip
    """
    # we are assuming a square image here
    # grid = size1 ** (1 / 2)
    # assert grid % 1 == 0, f"grid is {grid}"

    position_enc = torch.arange(0.0, dim1 * dim2).view(int(dim1), int(dim2))
    flipped_enc = torch.flip(position_enc, dims=[1])
    dists_pos = torch.cdist(position_enc.view([1, -1, 1]), flipped_enc.view([1, -1, 1]))
    correspondance = torch.where(dists_pos == 0, 1, 0).squeeze()
    return correspondance


def compute_triplet_loss(
    convs_1: torch.Tensor,
    convs_2: torch.Tensor,
    margin: float = 0.1,
    flip_corres: torch.Tensor = get_position_enc_flip(5, 5),
) -> torch.Tensor:
    """computes the distances under a transformation, only implemented for hor flip

    :param convs_1: [N, C, W, H] the computed conv features from the original features
    :param convs_2: [N, C, H, H] the computed conv features from the transformed features
    :param dist_type: which distance to each, either 'euclidean' or 'cosine', defaults to 'euclidean'
    :param transf_dict: dictionary containing which transformations have been applied to image
    :param exclude_middle: whether to exclude the middle from the correspondance matrix. If true, middle voxels
    are not considered
    :return: (loss_flip, loss_spatial)
        loss_flip: dependent on transformation, same image part but different location
        loss_spatial: independent on transformation, same spatial location but different image part
    """

    # reshape representations from [N, C, W, H] to [N, W*H, C]
    convs_1 = convs_1.view([convs_1.shape[0], convs_1.shape[1], -1]).swapaxes(1, 2)
    convs_2 = convs_2.view([convs_2.shape[0], convs_2.shape[1], -1]).swapaxes(1, 2)

    # compute distance between patch representations
    dists = torch.cdist(convs_1, convs_2)

    # correspondence matrix
    transf_correspondance = flip_corres.to(dists.device)

    # exclude middle row, as for these the pos and neg sample are the same
    diag_matrix = torch.diag(torch.ones(dists.shape[-1])).to(dists.device).detach()
    transf = torch.clip(transf_correspondance - diag_matrix, 0).detach()

    # Determine which of the anchors have a positive pair/match (after transformation)
    pos_pairs = torch.amax(transf, dim=1)

    # Get the positive distances (after transformation),
    # If there are multiple matches, take the mean (by dividing by sum), clamp(min=1) to prevent division by 0
    pos_dists = torch.sum(dists * transf.expand_as(dists), dim=-1) / torch.sum(transf, dim=1).clamp(min=1)

    # Mask out all the distances that do not have a positive pair
    masked_pos_matches = torch.masked_select(pos_dists, pos_pairs.expand_as(pos_dists) > 0).reshape(
        pos_dists.shape[0], -1
    )

    # if there are positive matches, compute triplet loss
    if masked_pos_matches.shape[1] != 0:
        # images that are same voxel but different image part after flipping (equiv of transf for no flip)
        spatial = torch.clip(diag_matrix - transf_correspondance, 0).detach()

        # Get the negative distances (from same pixels, but different image part)
        neg_dists = torch.sum(dists * spatial.expand_as(dists), dim=-1)

        # Mask out the distances that do not have a positive pair
        masked_neg_matches = torch.masked_select(neg_dists, pos_pairs.expand_as(pos_dists) > 0).reshape(
            dists.shape[0], -1
        )

        # compute triplet from positive and negative pairs
        single_triplets = torch.nn.functional.relu((masked_pos_matches - masked_neg_matches) + margin)

        triplet_loss = torch.mean(single_triplets, dim=1)
    else:
        # no positive matches, so we can't compute triplet loss
        triplet_loss = torch.Tensor([0])

    return triplet_loss


class RegMetricLoss_Protos(nn.Module):
    def __init__(
        self,
        sigma: float = 1.0,
        w_leak: float = 0.2,
        distance: str = "euclidean",
    ) -> None:
        super(RegMetricLoss_Protos, self).__init__()
        self.sigma = sigma
        self.w_leak = w_leak
        self.alpha: Optional[float] = None
        self.distance = distance

    def forward(self, distances: torch.Tensor, label: torch.Tensor, model: ProtoNet) -> torch.Tensor:

        label_samples = label.unsqueeze(1)
        label_prototypes = model.proto_classes.unsqueeze(0)  # type:ignore

        if len(distances.shape) != len(label_samples.shape):
            label_samples = label.view([label.shape[0], 1, 1, 1])
            label_prototypes = model.proto_classes.view([1, distances.shape[1], 1, 1])  # type: ignore

        # label distance
        l_dist = torch.abs(label_samples - label_prototypes)

        # weight of each prototype
        w = (torch.exp(-((l_dist / self.sigma) ** 2) / 2) + self.w_leak).detach()

        # label vs distance difference
        diff = torch.abs(distances - l_dist)

        # compute loss
        w_diff = w * diff
        loss = w_diff.sum() / (w.sum() + 1e-9)

        return loss


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.1,
        squared: bool = False,
        hard_mining: bool = False,
        proto_dims: list[int] = [5, 5]
    ) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        self.hard_mining = hard_mining

        print("initialization of flip encoding is hardcoded")
        self.flip_corres = get_position_enc_flip(proto_dims[0], proto_dims[1])

    def forward(self, convs_1: torch.Tensor, convs_2: torch.Tensor) -> torch.Tensor:

        triplet_loss = compute_triplet_loss(
            convs_1,
            convs_2,
            margin=self.margin,
            flip_corres=self.flip_corres,
        )

        return torch.mean(triplet_loss)


class RegressionMetric_total(nn.Module):
    def __init__(self, coefs: LossCoefs, proto_dims: list = [5, 5]) -> None:
        super(RegressionMetric_total, self).__init__()

        self.coefs = coefs

        # Metric loss
        self.MetricLoss = RegMetricLoss_Protos(
            sigma=coefs.metric_sigma,
            w_leak=self.coefs.w_leak,
        )

        # Triplet loss
        self.TripletLoss = TripletLoss(
            margin=coefs.triplet_margin,
            proto_dims=proto_dims,
        )

    def forward(
        self,
        label: torch.Tensor,
        model: ProtoNet,
        forward_dict: dict,
        forward_dict_transf: Optional[dict] = None,
    ) -> dict:
        self.total_loss = 0
        self.losses_dict = {}

        # update metric loss
        if self.coefs.metric > 0:
            metricloss = self.MetricLoss(forward_dict["min_distances"], label, model)
            self.losses_dict["Metric"] = metricloss
            self.update_total_loss(metricloss * self.coefs.metric)

        # update triplet loss
        if self.coefs.triplet_loss != 0 and forward_dict_transf is not None:
            assert forward_dict_transf is not None
            tripletloss = self.TripletLoss(
                forward_dict["conv_features"],
                forward_dict_transf["conv_features"],
            )
            if tripletloss != 0:
                self.losses_dict["triplet_loss"] = tripletloss
                self.update_total_loss(self.losses_dict["triplet_loss"] * self.coefs.triplet_loss)

        self.losses_dict["total_loss"] = self.total_loss
        return self.losses_dict

    def update_total_loss(self, loss: torch.Tensor) -> None:
        self.total_loss = self.total_loss + loss  # type:ignore
