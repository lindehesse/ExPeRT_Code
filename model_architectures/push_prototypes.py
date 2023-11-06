from __future__ import annotations  # make sure to assess type checking as strings

# so that it doesn't complain about litmodelproto
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
from model_architectures.ExPeRT_arch import PPNet_wholeprotos
from typing import TYPE_CHECKING, List, Optional
from torch.utils.data import DataLoader
import logging

if TYPE_CHECKING:
    from ..lightning_module_ExPeRT import LitModelProto

txt_logger = logging.getLogger("pytorch_lightning")
matplotlib.use("Agg")


class PushPrototypes:
    def __init__(self, pl_model: LitModelProto):
        """Initializes pushproto class to push the prototypes in latent space to closest image patch

        :param pl_model:PL model containing class attributes 'ppnet', 'params', 'current_epoch', and 'device'
        :param class_projection: whether to only project on own class, defaults to True
        """

        # The network
        self.ppnet: PPNet_wholeprotos = pl_model.ppnet
        self.ppnet.eval()

        # parameters settings
        self.params = pl_model.params
        self.current_epoch = pl_model.current_epoch
        self.proto_shape = self.ppnet.proto_shape
        self.num_proto = self.ppnet.num_prototypes

        # device to push protos on
        self.device = pl_model.device

        # Make save folder
        self.proto_epoch_dir = self.params.save_path_ims / "prototypes" / f"epoch_{self.current_epoch}"
        self.proto_epoch_dir.mkdir(parents=True, exist_ok=True)

    def push_prototypes(self, dataloader: DataLoader) -> None:
        """Pushes the model prototypes to closest image from dataloader

        :param dataloader: iterator over the data
        """

        self.init_save_variables()

        # Iterate over data, and each time push protoypes to closest image in batch
        if dataloader.batch_size is not None:
            for (batch_im, _, batch_names) in tqdm(dataloader):
                self.update_protos_batch(batch_im.to(self.device), batch_names)

        # Set everything in model
        self.update_model_variables()

        # save the filenames of the prototypes to a text file
        self.save_filenames_totext()
        self.ppnet.protonames = self.protonames

    def init_save_variables(self) -> None:
        """initializes tensors for the prototype representations"""
        # Saves the closest distance seen so far
        self.global_mindist = np.full(self.num_proto, np.inf)

        # saves the latent representation of prototype that gives the current smallest distance
        self.proto_latent_repr = np.zeros(self.proto_shape)

        # Proto information
        self.prototype_images = np.zeros(
            (self.num_proto, self.ppnet.img_size[1], self.ppnet.img_size[2], self.ppnet.img_size[0]), dtype=np.uint8
        )

        # Keep track of prototype classes if not class specfici
        self.protonames: List[Optional[str]] = [None] * self.num_proto

    def update_protos_batch(self, batch_im: torch.Tensor, batch_names: List[str]) -> None:
        """Update prototypes with batch of data

        :param batch_im: tensor containing the image data
        :param batch_names: list of the image names in the batch
        """

        # Put batch through trained network
        with torch.no_grad():
            if self.params.network_params.avg_pool:
                batch_convfeatures, batch_protodist = self.ppnet.avg_pool_distances(batch_im)
            else:
                batch_convfeatures, batch_protodist = self.ppnet.prototype_distances_ot(batch_im)

        # Get them to cpu
        batch_convfeatures = batch_convfeatures.detach().cpu().numpy()
        batch_protodist = batch_protodist.detach().cpu().numpy()

        # get all minimum distances to prototypes and corresponding indices
        proto_first = batch_protodist.swapaxes(0, 1)  # [num_proto, batchsize, ... ]
        dist_perproto = proto_first.reshape(self.num_proto, -1)  # [num_proto, dists]

        # get minimum distances (and corersponding indices) to each prototype
        mindist = np.amin(dist_perproto, axis=1)
        mindist_index = np.argmin(dist_perproto, axis=1)  # index of min distances [num_proto, ]
        index_unravel: tuple = np.unravel_index(mindist_index, proto_first.shape[1:])  # tuple of unraveled indices

        # check that the index unraveling is correct
        assert proto_first[0, index_unravel[0][0], index_unravel[1][0], index_unravel[2][0]] == mindist[0]

        # get all batch images corresponding to min distances
        whole_batch_im = batch_im[index_unravel[0]]

        # replace prototype if distance is smaller than before
        for proto_j in range(self.num_proto):
            if self.global_mindist[proto_j] > (newdist := mindist[proto_j]):
                # patch image prototypes
                batch_index = (index_unravel[0][proto_j], index_unravel[1][proto_j], index_unravel[2][proto_j])

                self.global_mindist[proto_j] = newdist
                if batch_convfeatures.shape[2:] == batch_protodist.shape[2:]:
                    latent_repr = batch_convfeatures[batch_index[0], :, batch_index[1], batch_index[2]]
                    self.proto_latent_repr[proto_j] = latent_repr.reshape(-1, 1, 1)
                else:
                    latent_repr = batch_convfeatures[batch_index[0], :, :, :]
                    self.proto_latent_repr[proto_j] = latent_repr

                self.prototype_images[proto_j] = np.transpose(whole_batch_im[proto_j].cpu().numpy(), (1, 2, 0)) * 255
                self.protonames[proto_j] = batch_names[batch_index[0]]

    def update_model_variables(self) -> None:
        """updates the model with the new prototype representations"""
        self.ppnet.prototype_vectors.copy_(
            torch.tensor(self.proto_latent_repr, dtype=torch.float32, requires_grad=True)
        )
        self.ppnet.prototype_images = torch.tensor(self.prototype_images)

    def save_filenames_totext(self) -> None:
        """Save the prototype filenames to a txt file"""
        # Save the filenames that were used to push
        datafile_path = Path(self.proto_epoch_dir) / "Prototype_Information_files.txt"
        with open(datafile_path, "w") as filehandle:
            for listitem in self.protonames:
                filehandle.write("%s\n" % listitem)
