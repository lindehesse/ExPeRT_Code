from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, Undefined
from enforce_typing import enforce_types
from pathlib import Path
from typing import List, Optional

MRI_datasets = ["IXI"]
FeatureBackbones = ["resnet18"]


@enforce_types
@dataclass_json
@dataclass
class OptimParams:
    # All parameters regarding optimization and lr decay
    joint_optim_lr: float = 5e-4
    joint_lr_stepsize: int = 50
    joint_lr_decay: bool = True
    emb_s_lr: float = 0.01


@dataclass_json
@dataclass
class LossCoefs:
    # Coefficients for the loss calculation
    metric: float = 1
    w_leak: float = 0.05
    metric_sigma: float = 1.0
    triplet_loss: float = 10.0
    triplet_margin: float = 0.1


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class NetworkParams:
    # Size of the input images, initialised from general parameters
    img_size: List[int] = field(default_factory=lambda: [1, 192, 160])  # required to set prototype shape correctly
    
    # which backbone, only implemented resnet18
    base_architecture: str = "resnet18"

    # whether to use a pretrained backbone
    pretrained: Optional[str] = "imagenet"  # can be imagenet or None

    # whether to add additional layers
    use_add_on_layers: bool = True

    # activation function of the last layer of the feature extractor (before additiona llayers), can be relu or sigmoid
    last_act: str = "relu"  

    # init r value during eval (testing)
    prediction_r_init: float = 5.0

    # parameters for the prototype-layer
    num_prototypes: int = 100
    init_emb_s: float = 10.0
    num_channels: int = 512

    # shape of prototypes
    proto_shape: List[int] = field(default_factory=list)

    # range of prototype labels (set at start of training)
    proto_range : List[int] = field(default_factory=list[int])

    # how to compute distances
    use_OT: bool = True  # whether to use optimal transport (else diagonals are taken)
    avg_pool: bool = False

    def set_prototype_shape(self) -> None:
        """Set the prototype shape based on the image size"""
        self.proto_shape = [
            int(self.num_prototypes),
            int(self.num_channels),
            int(self.img_size[1] / 32),
            int(self.img_size[2] / 32),
        ]


@enforce_types
@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class Parameters:
    # datapaths
    data_path: str
    save_basepath: str
    datasplit_path: str

    # Name for experiment
    experiment_run: str = "MRI_Age"
    run_name: str = "Run1"

    # Define crossvalidation
    cv_fold: int = 0

    # Define network parameters
    img_size: List[int] = field(default_factory=lambda: [1, 192, 160])

    # Define training parameters
    num_train_epochs: int = 300  
    push_epochs: List[int] = field(default_factory=lambda: [0, 75, 100, 125, 150, 175, 200, 225, 250, 275, 299])

    # Define batchsizes
    train_batch_size: int = 30
    val_batch_size: int = 30
    push_batch_size: int = 30

    # which 2D slice indices to use from 3D volumes
    slice_indices_train: List[int] = field(default_factory=lambda: [75, 76, 77, 78, 79, 80, 81, 82, 83, 84])
    slice_indices_test: List[int] = field(default_factory=lambda: [79])

    # Define folders names for saving (see to True for saving prototypes)
    save_protodir: Path = Path("prototypes")
    save_prototypes: bool = False

    # Define dataset
    dataset_name: str = "IXI"
    preload_data: bool = True

    # Define optimizers and loss coefs
    optim_params: OptimParams = OptimParams()
    loss_coefs: LossCoefs = LossCoefs()
    network_params: NetworkParams = NetworkParams()

    # Initialize variables defined in post_init and make savepaths
    train_dir: Path = field(default_factory=Path)
    test_dir: Path = field(default_factory=Path)
    val_dir: Path = field(default_factory=Path)
    val_labels: Path = field(default_factory=Path)
    train_labels: Path = field(default_factory=Path)
    test_labels: Path = field(default_factory=Path)

    # these are generated from the save_basepath
    save_path_experiment: Path = field(default_factory=Path)
    save_path: Path = field(default_factory=Path)
    save_path_ims: Path = field(default_factory=Path)

    def __post_init__(self) -> None:
        # Create the correct prototype shape based on the image size
        self.network_params.img_size = self.img_size
        self.network_params.set_prototype_shape()

    def set_savepaths(self) -> None:
        # Make full folder names for saving
        self.save_path_experiment = Path(self.save_basepath) / self.experiment_run
        self.save_path = self.save_path_experiment / self.run_name
        self.save_path_ims = self.save_path / "img"
        self.save_path_ims.mkdir(parents=True, exist_ok=True)
