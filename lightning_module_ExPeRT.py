import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_info
import math
import logging
import time
import torchmetrics
from collections import Counter
from torch.utils.data import DataLoader
from typing import Optional

import model_architectures.ExPeRT_arch as model
from model_architectures.ExPeRT_arch import PPNet_wholeprotos
from loss_function import (
    RegressionMetric_total,
)
from model_architectures.push_prototypes import PushPrototypes
from define_parameters import Parameters

txt_logger = logging.getLogger("pytorch_lightning")


class LitModelProto(pl.LightningModule):
    def __init__(self, params: Parameters, dataloader_push: DataLoader) -> None:
        super().__init__()
        self.params = params

        # Initialize network
        self.ppnet: PPNet_wholeprotos = model.PPNet_wholeprotos(network_params=params.network_params)

        # Set loss
        self.protoloss = RegressionMetric_total(self.params.loss_coefs, self.params.network_params.proto_shape[2:])
        self.dataloader_push = dataloader_push

        # Set to false in order to work with multiple optimizers
        self.automatic_optimization = False

        # Set up metrics
        self.metrics = ["MAE"]

        if (metricname := "MAE") in self.metrics:
            for datatype in ["val", "test"]:
                setattr(self, f"{datatype}_{metricname}", torchmetrics.MeanAbsoluteError())

            # make seperate one for push validation epochs
            setattr(self, f"val_{metricname}_push", torchmetrics.MeanAbsoluteError())

        # Init training stage parameter
        self.push_epoch: bool = False
        self.first_batch: bool = True
        self.push_history: list[int] = []

        self.save_hyperparameters("params")

        if self.logger is not None:
            assert isinstance(self.logger, MLFlowLogger)
            self.logger: MLFlowLogger

    def forward(  # type:ignore
        self, batch: tuple, step: Optional[str] = None
    ) -> dict:
        """Forward a batch through the network
        Args:
            batch (tuple): batch of data containing ims and labels
        Returns:
            dictionary with forward pass results
        """

        # Forward batch through network
        batch_ims = batch[0].to(self.device)

        # forward pass (with train less is computed so a bit faster for trianing)
        train = True if step == "train" else False
        if self.params.network_params.avg_pool:
            forward_dict = self.ppnet(batch_ims, use_avgpool=True, train=train)
        else:
            forward_dict = self.ppnet(batch_ims, train=train, current_epoch=self.current_epoch)

        return forward_dict

    def predict_step(self, batch: tuple, batch_idx: int) -> tuple:  # type:ignore
        """_summary_

        :param batch: _description_
        :param batch_idx: _description_
        :return: _description_
        """
        input = batch[0]
        label = batch[1]
        filenames = batch[2]

        forward_dict = self.forward(batch)
        return (input, label, forward_dict, filenames)

    def forward_step(self, batch: tuple, step: str = "train", batch_idx: int = 1) -> dict:  # type:ignore
        """Defines forward step that can be used for validation and training

        Args:
            batch (tuple): tensor, labels

        Returns:
            dict: dictionary containing all losses
        """
        # Extract image and label from batch
        input = batch[0]
        label = batch[1]
        image_paths = batch[2]

        # Forward batch through network
        assert list(input.shape[1:]) == self.params.img_size

        # forward pass
        forward_dict = self.forward(batch, step=step)

        # Compute consistency transformations if required
        if self.params.loss_coefs.triplet_loss > 0:
            transf_input = torch.flip(input, dims=[-1])
            forward_dict_transf = self.ppnet(transf_input, current_epoch=self.current_epoch)
        else:
            forward_dict_transf = None

        # Compute losses
        losses_dict = self.protoloss(label, self.ppnet, forward_dict, forward_dict_transf=forward_dict_transf)

        if step == "train":
            return {
                **losses_dict,
                "target": label.detach().float(),
                "image_names": image_paths,
            }
        else:
            return {
                **losses_dict,
                "pred": forward_dict["logits_raw"].detach(),
                "target": label.detach().float(),
                "convs_features": forward_dict["conv_features"].detach(),
                "image_names": image_paths,
            }

    def training_step(self, batch: tuple, batch_idx: int) -> dict:  # type:ignore
        """Defines the training step for a single batch
        Args:
            batch (tuple): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """

        joint = self.optimizers()  # type:ignore
        lr_sched = self.lr_schedulers()

        # Do forward step
        losses_dict = self.forward_step(batch, step="train", batch_idx=batch_idx)
        loss = losses_dict["total_loss"]

        # Log here per step (do not log target)
        temp_dict = dict(losses_dict)
        temp_dict.pop("target")
        temp_dict.pop("image_names")

        self.log_dict({f"train_step_{k}": v for k, v in temp_dict.items()})

        #  Joint training stage
        joint.zero_grad()  # type:ignore
        self.manual_backward(loss)
        joint.step()  # type:ignore

        # Do lr step once per epoch
        if lr_sched is not None:
            if self.trainer.is_last_batch:
                if self.params.optim_params.joint_lr_decay:
                    lr_sched.step()  # type:ignore
        else:
            raise ValueError("Training stage not recognized")

        self.trainer.fit_loop.running_loss.append(loss)

        return {(f"{k}"): (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in losses_dict.items()}

    def validation_step(  # type:ignore
        self, batch: tuple, batch_idx: int
    ) -> dict:
        """Defines the validation step for a single batch
        Args:
            batch (tuple): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """

        return self.forward_step(batch, step="val", batch_idx=batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> dict:  # type:ignore
        """Defines the test step for a single batch
        Args:
            batch (tensor): batch of images
            batch_idx (int): batch no.

        Returns:
            dict: dictionary with all loss elements
        """
        return self.forward_step(batch, step="test", batch_idx=batch_idx)

    def validation_step_end(self, outputs: dict) -> None:  # type:ignore
        """Logging for validation step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        predictions = outputs["pred"]
        predictions_nonans = predictions[~predictions.isnan()]
        targets_matched = outputs["target"][~predictions.isnan()]

        if len(predictions_nonans) > 0:
            for metric in self.metrics:
                getattr(self, f"val_{metric}")(predictions_nonans, targets_matched)

            if self.push_epoch:
                # log for pushing also value
                getattr(self, "val_MAE_push")(predictions_nonans, targets_matched)

    def test_step_end(self, outputs: dict) -> None:  # type:ignore
        """Logging for testing step

        Args:
            outputs (dict): dict with keys 'pred' and 'target'
        """
        predictions = outputs["pred"]
        predictions_nonans = predictions[~predictions.isnan()]
        targets_matched = outputs["target"][~predictions.isnan()]

        if len(predictions_nonans) > 0:
            for metric in self.metrics:
                getattr(self, f"test_{metric}")(predictions_nonans, targets_matched)

    def log_epoch_end(self, output_losses: dict, tag: str = "train"):  # type:ignore
        """Logs all the losses at end of epoch

        Args:
            output_losses (dict): losses returned by train step
            tag (str, optional): Whether we are logging from a train or validation epoch. Defaults to 'train'.
        """
        # Log lr per epoch
        if tag == "train":
            lr_sched = self.lr_schedulers()
            if lr_sched is not None:
                lrates = lr_sched.get_last_lr()  # type:ignore
                for i, lr in enumerate(lrates):
                    self.logger.experiment.log_metric(
                        self.logger.run_id,
                        key=f"lr_epoch_{i}",
                        value=lr,
                        step=self.current_epoch,
                    )

            # log parameter.r for
            self.logger.experiment.log_metric(
                self.logger.run_id,
                key="emb_loss_s",
                value=self.ppnet.emb_loss_s,
                step=self.current_epoch,
            )

        if (tag == "val") or (tag == "test"):
            # Log the metrics at end of epoch
            for metric in self.metrics:
                metric_function = getattr(self, f"{tag}_{metric}")

                if metric_function._update_count == 0:
                    continue
                else:
                    metric_value = metric_function.compute()
                    metric_function.reset()
                    if self.logger is not None:
                        self.logger.experiment.log_metric(
                            self.logger.run_id,
                            key=f"{tag}_epoch_{metric}",
                            value=metric_value.item(),
                            step=self.current_epoch,
                        )

                    print(f"{metric}: {metric_value}")

        # Sum total loss over all steps
        total_counters: Counter = Counter()
        for x in output_losses:
            x.pop("target")
            x.pop("image_names")
            if tag != "train":
                x.pop("convs_features")
                x.pop("pred")
            total_counters.update(Counter(x))

        for key in total_counters:
            self.logger.experiment.log_metric(
                self.logger.run_id,
                key=f"{tag}_epoch_{key}",
                value=total_counters[key] / len(output_losses),
                step=self.current_epoch,
            )

        # Log to monitoring loss
        self.average_loss = total_counters["total_loss"] / len(output_losses)
        self.log(f"{tag}_loss_monitor", self.average_loss, sync_dist=True)

        # Log the metric loss
        self.metric_loss = total_counters["Metric"] / len(output_losses)
        self.log(f"{tag}_metric_monitor", self.metric_loss, sync_dist=True)

        # log best loss after push seperately
        if tag == "val":
            if self.push_epoch:
                metric_function = getattr(self, "val_MAE_push")
                MAE_push = metric_function.compute()
                metric_function.reset()
                self.logger.experiment.log_metric(
                    self.logger.run_id,
                    key=f"{tag}_epoch_MAE_push",
                    value=MAE_push.item(),
                    step=self.current_epoch,
                )
                # required for callbacks
                self.log("val_loss_afterpush", self.average_loss, sync_dist=True)
                self.log("val_metric_afterpush", self.metric_loss, sync_dist=True)
            else:
                self.log("val_loss_afterpush", math.inf, sync_dist=True)
                self.log("val_metric_afterpush", math.inf, sync_dist=True)

        # Log to txt logger
        local_time = time.ctime(time.time())
        rank_zero_info(local_time + f" Total {tag} loss: {self.average_loss:.4f}")

    def training_epoch_end(self, training_steps_output: dict) -> None:  # type:ignore
        """Function to collect losses from epoch and save this per epoch

        Args:
            training_steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(training_steps_output, tag="train")

    def validation_epoch_end(  # type:ignore
        self, validation_steps_output: dict
    ) -> None:  # type:ignore
        """Function to collect losses from epoch and save this per epoch

        Args:
            validation_steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(validation_steps_output, tag="val")

    def test_epoch_end(self, steps_output: dict) -> None:  # type:ignore
        """Function to collect losses from testing epoch and save this per epoch

        Args:
            steps_output (dict): dict containing all losses
        """
        self.log_epoch_end(steps_output, tag="test")

    def configure_optimizers(self) -> tuple:  # type:ignore
        """Set up all optimizers for training

        Returns:
            list, list: optimizers, lr_schedulers
        """

        optim_params = self.params.optim_params

        # Joint training optimizers and lr schedulers
        joint_optim_specs = [
            {"params": self.ppnet.features.parameters()},
            {"params": self.ppnet.prototype_vectors},
            {"params": self.ppnet.emb_loss_s, "lr": optim_params.emb_s_lr},
        ]

        if hasattr(self.ppnet, "add_on_layers"):
            joint_optim_specs.append({"params": self.ppnet.add_on_layers.parameters()})

        joint_optim = torch.optim.Adam(joint_optim_specs, lr=optim_params.joint_optim_lr)  # type:ignore
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            joint_optim, step_size=optim_params.joint_lr_stepsize, gamma=0.5
        )

        # Gather all optimizers together and return
        optimizers = [joint_optim]
        lr_schedulers = {"scheduler": joint_lr_scheduler, "name": "joint_scheduler"}

        return optimizers, lr_schedulers

    def on_validation_epoch_start(self) -> None:  # type:ignore
        # At the end of the joint training stage, push prototype vectors to closest image patch
        # push history avoids pushing twice in zeroth epoch (because of pl)
        if self.current_epoch in self.params.push_epochs and self.current_epoch not in self.push_history:
            # Push prototypes
            txt_logger.info("Prototype_Pushing")
            pushprototypes = PushPrototypes(self)
            pushprototypes.push_prototypes(self.dataloader_push)

            # Log new prototypes to mlflow
            self.logger.experiment.log_artifacts(
                self.logger.run_id,
                local_dir=pushprototypes.proto_epoch_dir,
                artifact_path=f"prototypes/epoch{self.current_epoch}",
            )
            # self.save_model("after_protopushing")
            txt_logger.info("warning: model is not saved after prototype pushing")

            self.push_epoch = True
            self.push_history.append(self.current_epoch)
        else:
            self.push_epoch = False

    @rank_zero_only
    def save_model(self, savename: str) -> None:
        path = Path(self.params.save_path) / "saved_models"
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.ppnet.state_dict(), path / f"Epoch_{self.current_epoch}_{savename}.pth")
