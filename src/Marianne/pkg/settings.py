import dataclasses as _dataclasses
import os as _os
import socket as _socket
import warnings as _warnings
from pathlib import Path as _Path

import torch

#######################################################
# device setting


def get_torch_device():
    """Auto-detect CUDA, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        print("✓ Using CUDA GPU")
        return "cuda"
    if torch.backends.mps.is_available():
        print("✓ Using Apple MPS GPU")
        return "mps"
    print("✓ Using CPU")
    return "cpu"


#######################################################
# directory setting


def cache_dir() -> _Path:
    """Return a directory to use as cache."""
    cache_path = _Path(_os.environ.get("CACHE_DIR", _Path.home() / ".cache" / __package__))
    if not cache_path.is_dir():
        cache_path.mkdir(exist_ok=True, parents=True)
    return cache_path


def _assert_dir(dir_location: _Path) -> None:
    """Check if directory exists & throw warning if it doesn't."""
    if not dir_location.is_dir():
        _warnings.warn(
            f"Using non-existent directory: {dir_location}\nCheck your settings & env variables.",
            stacklevel=1,
        )


#######################################################
# Path Settings


@_dataclasses.dataclass
class _MyPathSettings:
    PROJECT_NAME: str = "Conv-behavior-annotator"

    def __post_init__(self) -> None:
        hostname = _socket.gethostname()
        prefix = _Path("/lustre/fswork/projects/rech/eqb/commun") if "Jean-Zay" in hostname else _Path("/scratch2/jliu")
        self.DATA_DIR = (prefix / "Feedback").resolve()

        if not self.DATA_DIR.is_dir():
            _warnings.warn(
                f"Resolved DATA_DIR does not exist: {self.DATA_DIR}",
                stacklevel=1,
            )

    # ---- Shared project layout ----

    @property
    def dataset_root(self) -> _Path:
        path = self.DATA_DIR / "datasets"
        _assert_dir(path)
        return path

    @property
    def script_dir(self) -> _Path:
        return self.DATA_DIR / self.PROJECT_NAME

    @property
    def model_dir(self) -> _Path:
        return self.DATA_DIR / "models"

    @property
    def result_dir(self) -> _Path:
        return self.DATA_DIR / "results"


#######################################################
# Instance of Settings

PATH = _MyPathSettings()
