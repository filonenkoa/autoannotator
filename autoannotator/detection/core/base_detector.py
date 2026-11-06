import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from typing import List, Sequence
from autoannotator.types.base import Detection
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import (
    attempt_download_onnx,
    attempt_download_custom_op,
    get_project_root,
)
from autoannotator.custom_ops import custom_ops_mapper, custom_ops_storage_path

_ROOT = get_project_root()


class BaseDetector(ABC):
    """
    This is a base abstract detector class

    Arguments:
        config (DetectionConfig): detector config
    """

    def __init__(self, config: DetectionConfig):
        self.config = config

        self.session = None
        self._init_session()

    def __repr__(self):
        cls_name = self.__class__.__name__
        kwargs = ", ".join([f"{k}={v}" for k, v in self.config.dict().items()])
        return f"{cls_name}({kwargs})"

    def __call__(self, img: np.ndarray) -> Sequence[Detection]:
        """
        Detect objects on image

        Arguments:
            img (np.ndarray): The input RGB image, HxWx3

        Returns:
            List[Detection]: List of detected objects
        """
        out = self._predict(img)
        return out

    def predict_batch(self, imgs: List[np.ndarray]) -> Sequence[Sequence[Detection]]:
        """Detect objects on a batch of images.

        Default implementation runs per-image inference in a Python loop.
        Subclasses can override to use true batched ONNXRuntime inference.
        """
        return [self._predict(img) for img in imgs]

    def _init_session(self):
        """Init onnx runtime session"""
        if self.session is None:
            import onnxruntime

            if self.config.device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                providers = ["CUDAExecutionProvider"]

            assert self.config.weights is not None
            if not Path(self.config.weights).is_file():
                attempt_download_onnx(self.config.weights, self.config.url)
                # raise FileNotFoundError(f'No onnx weights found at {self.config.weights}')

            options = onnxruntime.SessionOptions()
            if len(self.config.onnx_custom_ops_libraries) > 0:
                for lib in self.config.onnx_custom_ops_libraries:
                    local_file_path = Path(
                        _ROOT, custom_ops_storage_path, lib
                    ).expanduser()
                    custom_ops_url = custom_ops_mapper.get(lib, None)
                    if not custom_ops_url:
                        raise Exception(
                            f"There is no known URL for {self.config.onnx_custom_ops_libraries} custom operations shared library"
                        )
                    attempt_download_custom_op(local_file_path, custom_ops_url)
                    assert Path(local_file_path).is_file()
                    if local_file_path.suffix == ".so":
                        options.register_custom_ops_library(local_file_path.as_posix())
            self.session = onnxruntime.InferenceSession(
                self.config.weights, options, providers=providers
            )

    @property
    @abstractmethod
    def name(self):
        """Return detector name"""
        raise NotImplementedError

    @abstractmethod
    def _predict(self, img: np.ndarray) -> Sequence[Detection]:
        raise NotImplementedError
