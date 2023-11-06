import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

from define_parameters import NetworkParams
from model_architectures.optimal_transport_helpers import batched_optimal_transport_log
from model_architectures import resnet_features

NetworkFeatures = type[resnet_features.ResNet_features]


class PPNet_wholeprotos(nn.Module):
    def __init__(
        self,
        network_params: NetworkParams = NetworkParams(),
        init_weights: bool = True,
    ):
        super(PPNet_wholeprotos, self).__init__()

        self.img_size = network_params.img_size
        self.num_prototypes = network_params.num_prototypes
        self.proto_shape = network_params.proto_shape

        self.use_add_on_layers = network_params.use_add_on_layers
        self.last_act = network_params.last_act
        self.proto_range = (network_params.proto_range[0], network_params.proto_range[1])

        self.output_num = 1
        self.current_epoch = 0

        self.set_features_backbone(network_params)

        # This has to be at end to make initialization reproducible
        if self.use_add_on_layers:
            self.create_add_on_layers()
            if init_weights:
                self._initialize_weights()

        self.create_prototypes()

        # Required for l2 convolution
        self.ones = nn.Parameter(torch.ones(self.proto_shape), requires_grad=False)

        self.r = nn.Parameter(torch.tensor([network_params.prediction_r_init]), requires_grad=False)

        self.num_patches = self.output_size_conv[0] * self.output_size_conv[1]
        self.emb_loss_s = nn.Parameter(torch.ones((1)) * network_params.init_emb_s, requires_grad=True)

    def set_features_backbone(self, network_params: NetworkParams) -> None:
        """sets the feature backbone of the network (i.e. resnet18)

        :param network_params: the dataclass containing the network parameters
        """
        # keeps only the letters (i.e. resnet)
        featuretype = re.sub(r"[^a-zA-Z]", "", network_params.base_architecture)
        module_str = f"{featuretype}_features"

        # gets the module and function by their string value
        module = globals()[module_str]
        base_architecture = getattr(module, f"{network_params.base_architecture}_features")

        features: NetworkFeatures = base_architecture(
            pretrained=network_params.pretrained, in_channels=self.img_size[0], last_act=self.last_act
        )
        self.features = features

    def create_prototypes(self) -> None:
        """Creates the prototypes"""

        # Determine the output size of the conv layers
        dummy_data = torch.zeros((1, *self.img_size))
        outconv = self.features(dummy_data)

        if self.use_add_on_layers:
            outconv = self.add_on_layers(outconv)
        self.output_size_conv: torch.Tensor = outconv.shape[2:]

        # Create prototypes and prototype classes
        self.prototype_vectors = nn.Parameter(torch.rand(self.proto_shape), requires_grad=True)
        proto_classes = torch.linspace(self.proto_range[0], self.proto_range[1], int(self.num_prototypes))

        # Save all proto info
        self.register_buffer("proto_classes", proto_classes.float())
        self.register_buffer(
            "prototype_images",
            torch.zeros(self.num_prototypes, self.img_size[1], self.img_size[2], self.img_size[0], dtype=torch.uint8),
        )

    def create_add_on_layers(self) -> None:
        """Creates two additional layers at the end of the feature backbone

        :raises Exception: if the feature backbone is not known
        """
        # Determine number of channels of last feature layer ( = num input channels add layers)
        first_add_on_layer_in_channels = [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][
            -1
        ].out_channels

        # Create the layer block of activations
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels,
                out_channels=self.proto_shape[1],
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.proto_shape[1], out_channels=self.proto_shape[1], kernel_size=1),
        )

        # Add the last activation layer (typically sigmoid)
        self.add_on_layers.add_module("sigmoid", nn.Sigmoid())

    def _initialize_weights(self) -> None:
        """Initializes the additional layer weights in the network"""
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def prototype_distances_ot(self, x: torch.Tensor, return_ot: bool = False) -> tuple:
        """Generates distances between samples and prototypes usning OT
        :param x: the batch
        :param return_ot: whether to return to OT flow matrix, defaults to False
        :return: tuple of:
            conv features [B, C, H, W] (torch.Tensor),
            distances [B, no. prototypes] (torch.Tensor),
            OT flow [B, no. prototypes, n_patches, n_patches] (torch.Tensor)
        """

        conv_features = self.conv_features(x)

        # move channel dimension to end and flatten other dims
        features = torch.einsum("bchw-> bhwc", conv_features)

        # do same for prototypes
        prototypes = torch.einsum("bchw-> bhwc", self.prototype_vectors)
        prototypes_res = prototypes.reshape(-1, prototypes.shape[-1])

        assert (
            prototypes.shape[1:] == features.shape[1:]
        ), "prototypes and conv_features must have same shape. Check image size in parameter files"

        distances, OT_flow, image_dist, proto_dist, patch_distances = self.compute_OT_distances(
            conv_features,
            prototypes_res,
            shape1=(features.shape[0], features.shape[1] * features.shape[2]),
            shape2=(self.prototype_vectors.shape[0], self.prototype_vectors.shape[2] * self.prototype_vectors.shape[3]),
        )

        if return_ot:
            return conv_features, distances, OT_flow, image_dist, proto_dist, patch_distances
        else:
            return conv_features, distances

    def compute_OT_distances(
        self,
        conv_features: torch.Tensor,
        prototypes_res: torch.Tensor,
        shape1: tuple[int, int],
        shape2: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # extract the patch distances
        features = torch.einsum("bchw-> bhwc", conv_features)
        features_res = features.reshape(-1, features.shape[-1])

        # compute OT distances
        patch_distances = torch.cdist(features_res, prototypes_res, p=2)
        patch_distances = patch_distances.view([shape1[0], shape1[1], shape2[0], shape2[1]])
        patch_distances_res = torch.einsum("bipj->bpij", patch_distances)

        # Get initial distribution for x
        m = patch_distances_res.shape[-2]

        # u needs to be of size [B, P, num_conv_patches], v of size [B, P, num_protopatches]
        image_dist, proto_dist = self.get_initial_distributions(m)

        # compute optimal transport
        OT, _ = batched_optimal_transport_log(patch_distances_res, image_dist, proto_dist, reg=0.1)

        # compute distances and scale them with the scaling factor (emb_loss_s)
        distances = (torch.sum(OT * (patch_distances_res), dim=(2, 3))[:, :, None, None]) * self.emb_loss_s

        return distances, OT, image_dist, proto_dist, patch_distances_res

    def get_initial_distributions(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """generate the initial distributions for the optimal transport matching
        :param m: length of the vector to generate
        :raises ValueError: returns error if initial_distribution_type is not implemented
        :return: initial distribution
        """

        # use uniform distribution
        dist_x = torch.Tensor(np.array([1 / (m)] * m)).to(self.prototype_vectors.device)
        return dist_x, dist_x

    def conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """extracts the convolutional features from the image

        :param x: tensor containing the image data
        :return: features extracted from the image
        """
        x = self.features(x)

        if self.use_add_on_layers:
            x = self.add_on_layers(x)

        return x

    def avg_pool_distances(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute distances between sample and prototypes using average pooling to compress them in vectorsfirst

        :param x: batch of samples
        :return: tuple of conv features [B, C, H, W] (torch.Tensor), distances [B, no. prototypes] (torch.Tensor)
        """

        conv_features = self.conv_features(x)
        conv_avg = F.avg_pool2d(conv_features, kernel_size=conv_features.shape[2:])

        # pool spatial dimensions using avgpool
        proto_avg = F.avg_pool2d(self.prototype_vectors, kernel_size=self.prototype_vectors.shape[2:])
        image_distances = torch.cdist(conv_avg.squeeze(), proto_avg.squeeze(), p=2) * self.emb_loss_s

        return conv_avg, image_distances.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, use_avgpool: bool = False, train: bool = False, current_epoch: int = 0) -> dict:
        """forward pass of the network

        :param x: batch of data
        :param use_avgpool: whether to use avgpooling, defaults to False
        :param train: whether it is a training epoch, then less is computed, defaults to False
        :return: dictionary including predictions
        """
        self.current_epoch = current_epoch

        # compute distances with avgpool or with ot between patches
        if use_avgpool:
            conv_features, min_distances = self.avg_pool_distances(x)
            ot_flow = None
            image_weights = None
            proto_weights = None
            patch_distances = None
        else:
            (
                conv_features,
                min_distances,
                ot_flow,
                image_weights,
                proto_weights,
                patch_distances,
            ) = self.prototype_distances_ot(x, return_ot=True)

        # create dictionary to return values
        min_distances = min_distances.squeeze()
        info_dict = {
            "min_distances": min_distances,
            "conv_features": conv_features,
            "ot_flow": ot_flow,
            "image_weights": image_weights,
            "proto_weights": proto_weights,
            "patch_distances": patch_distances,
        }

        if not train:
            prediction_dict = self.compute_age_predictions(min_distances)
            info_dict.update(prediction_dict)

        return info_dict

    def compute_age_predictions(self, min_distances: torch.Tensor) -> dict:
        """Compute the actual age predictions using the computed min distances to prototypes

        :param min_distances: tensor containing the distances of size [B, no. prototypes]
        :return: dictionary with predictions
        """
        # cut of distances at self.r and generate gaussian weights
        clipped_dists = torch.where(min_distances > self.r, torch.inf, min_distances)
        w = torch.exp(-((clipped_dists) ** 2) / (2 * (self.r.to(min_distances.device) / 3) ** 2))

        # make prediction
        class_IDx = torch.unsqueeze(self.proto_classes, dim=0)  # type:ignore
        prediction_raw = torch.sum(class_IDx * w, dim=1) / torch.sum(w, dim=1)

        # check if prediction contains nans
        numnans = torch.isnan(prediction_raw).sum()

        if numnans > 0:
            # get closest prototype to each sample
            sum_w = torch.sum(w, dim=1)
            closest_proto_index = torch.argmin(min_distances, dim=1)
            closest_proto_label = self.proto_classes[closest_proto_index]  # type:ignore

            # if the weights of the prototypes are all 0, use the label of the closest prototype
            prediction = torch.where(sum_w == 0, closest_proto_label, prediction_raw)

            # update the weights accordingly (used for proto matching matrix)
            for i in range(sum_w.shape[0]):
                if sum_w[i] == 0:
                    w[i, closest_proto_index[i]] = 1
        else:
            prediction = prediction_raw

        return {
            "logits": prediction,
            "logits_raw": prediction_raw,
            "numnans": numnans,
            "patch_preds": class_IDx * w,
            "used_weights": w,
        }
