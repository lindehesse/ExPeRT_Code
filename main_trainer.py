import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import logging
from pytorch_lightning.utilities import rank_zero_info


from lightning_module_ExPeRT import LitModelProto
from datamodules.datamodule import MyDataModuleBrainDataset
from helpers import load_json
from define_parameters import Parameters


def train_runner(params: Parameters) -> None:
    """Function to run the network training

    :param params: dataclass containing all training parameters
    """

    # Set up ML Flow logger to monitor experimnets
    mlf_logger: MLFlowLogger = MLFlowLogger(experiment_name=params.experiment_run, run_name=params.run_name)

    # Choose dataset
    dataset: MyDataModuleBrainDataset = MyDataModuleBrainDataset(params)
    params.network_params.proto_range = [dataset.min_train_label, dataset.max_train_label]

    # Set up checkpoint callbacks ------------------------------------------------

    # Save based on lowest metric loss
    checkpoint_callback = ModelCheckpoint(
        filename="bestmodel_{epoch:02d}-{valloss:02d}",
        monitor="val_metric_monitor",
        dirpath=params.save_path / "checkpoints",
    )

    # Save based on metric loss after prototype projection
    checkpoint_callback_stage2 = ModelCheckpoint(
        filename="bestmodel_push_{epoch:02d}-{valloss:02d}",
        monitor="val_metric_afterpush",
        dirpath=params.save_path / "checkpoints",
    )

    all_callbacks = [checkpoint_callback, checkpoint_callback_stage2]

    # set up trainer ------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator="auto",
        logger=mlf_logger,
        max_epochs=params.num_train_epochs,
        callbacks=all_callbacks,  # type:ignore
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train the model -----------------------------------------------------------
    model = LitModelProto(params, dataloader_push=dataset.push_dataloader())
    trainer.fit(model, dataset)
    rank_zero_info("Finished Training Succesfully")

    # Test the model ------------------------------------------------------------
    trainer.test(model, dataloaders=dataset.test_dataloader(), ckpt_path=checkpoint_callback_stage2.best_model_path)
    rank_zero_info("Finished Testing Succesfully")

    # Log txt logger to mlflow
    mlf_logger.experiment.log_artifact(run_id=mlf_logger.run_id, local_path=params.save_path / "logfile.log")


def run_crossval(params: Parameters, run_name: str, total_folds: int = 3) -> None:
    """Run cross validation with the given parameters

    :param params: parameters containing all the settings
    :param run_name: name of the run
    :param num_folds: total number of folds to run, defaults to 3
    """

    for fold in range(0, total_folds):
        # Set crossvalidation
        params.cv_fold = fold

        # Reinitialize savepaths
        params.run_name = f"{run_name}_fold{fold}"
        params.set_savepaths()

        # Set handler to new file
        # logging for outside trainer object
        txt_logger = logging.getLogger("pytorch_lightning")
        filehandler = logging.FileHandler(params.save_path / "logfile.log")
        txt_logger.addHandler(filehandler)

        # Run training
        rank_zero_info(f"Start fold {fold}")
        train_runner(params)

        # Remove handler from logger
        txt_logger.removeHandler(filehandler)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Run ProtoPnet code")

    # savepath must be provided
    parser.add_argument(
    "--save_basepath", type=str, required=True, help="Path to the folder where the results will be saved"
    )

    # CMD arguments to define data (point to dummy paths now)
    parser.add_argument("--data_path", type=str, default='dummy_data/datafolder', help="Path to the data folder")
    parser.add_argument("--datasplit_path", type=str, required=False, default='dummy_data/datasplit.json', help="Path to the datasplit json")

    # name of parameter file
    parser.add_argument(
        "-p",
        "--param_jsonpath",
        type=str,
        required=False,
        default="protonet_complete.json",
        help="Define the parameter file used for training (example: config/params.json)",
    )

    run_name = 'testing_run'
    experiment_name = 'ExPeRT_exps'

    # Parse command line arguments
    args_dict = vars(parser.parse_args())
    params_dict = load_json(args_dict["param_jsonpath"])
    params_exp = {
        **params_dict,
        "experiment_run": experiment_name,
        "run_name": run_name,
        "data_path": args_dict["data_path"],
        "save_basepath": args_dict["save_basepath"],
        "datasplit_path": args_dict["datasplit_path"],
    }
    params = Parameters.from_dict(params_exp)  # type:ignore

    run_crossval(params, run_name=run_name, total_folds=1)
