import os
import sys
import torch
import warnings

# Support both package and direct script execution for _workflow_utils
try:
    from ._workflow_utils import set_workflow_state  # type: ignore
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from _workflow_utils import set_workflow_state
from nequip.utils import get_current_code_versions, RankedLogger
from nequip.utils.global_state import set_global_state, get_latest_global_state
from nequip.data.datamodule import NequIPDataModule
from nequip.train import NequIPLightningModule

from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
import hydra
import os
from typing import Final, List  


# pre-emptively set this env var to get the full stack trace for convenience
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = RankedLogger(__name__, rank_zero_only=True)

_REQUIRED_CONFIG_SECTIONS: Final[List[str]] = [
    "run",
    "data",
    "trainer",
    "training_module",
]