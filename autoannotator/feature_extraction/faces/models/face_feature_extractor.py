from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from autoannotator.feature_extraction.core.feature_extractor import BaseFeatureExtrator, FaceFeatureExtractorConfig
from autoannotator.utils.image_preprocessing import normalize_image, np2onnx
from autoannotator.utils.misc import attempt_download_onnx
from autoannotator.utils.onnx_model_handler import OnnxModelHandler

    
class FaceFeatureExtractor(BaseFeatureExtrator):
    """The base class for face feature extractors.
    The logic consists of 3 parts: preprocessing, inference, and postprocessing.
    It is expected that FeatureExtractorConfig contains URL to download ONNX weights if ONNX file is not available locally.
    """
    def __init__(self, config: FaceFeatureExtractorConfig) -> None:
        super().__init__(config)
        lib_root = Path(__file__).absolute().parent.parent.parent.parent
        self.onnx_path = Path(lib_root, self.config.onnx_path)
        attempt_download_onnx(self.onnx_path.as_posix(), self.config.url)
        assert self.onnx_path.is_file(), f"Could not find {self.onnx_path.as_posix()}"
        self.model = OnnxModelHandler(self.onnx_path.as_posix(), device=self.device)
    
    def _preprocess(self, image: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        """Arrange color channels and normalize the input image or batch.

        Args:
            image (np.ndarray | Sequence[np.ndarray]): Input image or collection of images.

        Returns:
            np.ndarray: Preprocessed tensor with shape (B, C, H, W).
        """

        mean_vals = self.config.normalize_mean
        std_vals = self.config.normalize_std
        norm_mean: tuple[float, float, float] = (
            float(mean_vals[0]),
            float(mean_vals[1]),
            float(mean_vals[2]),
        )
        norm_std: tuple[float, float, float] = (
            float(std_vals[0]),
            float(std_vals[1]),
            float(std_vals[2]),
        )

        def _prepare_single(img: np.ndarray) -> np.ndarray:
            prepared = img.copy()
            prepared = normalize_image(
                prepared,
                norm_mean,
                norm_std,
            )
            return np2onnx(prepared, color_mode=self.config.color_format)

        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                items: Iterable[np.ndarray] = (image[i] for i in range(image.shape[0]))
                tensors = [_prepare_single(item) for item in items]
                return np.concatenate(tensors, axis=0)
            return _prepare_single(image)

        if isinstance(image, Sequence) and not isinstance(image, (bytes, str)):
            tensors = [_prepare_single(np.asarray(item)) for item in image]
            return np.concatenate(tensors, axis=0)

        return _prepare_single(np.asarray(image))
    
    def _forward(self, image: np.ndarray) -> np.ndarray:
        """Perform inference
        
        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Features (embedding, descriptor)
        """
        embedding = self.model(image)
        return embedding
    
    def _postprocess(self, tensor: np.ndarray) -> np.ndarray:
        """Process the output of current model.

        Returns either a single embedding vector or a batch of embeddings
        depending on the model output shape.
        """
        if tensor.ndim == 1:
            return tensor

        if tensor.ndim >= 2 and tensor.shape[0] == 1:
            return tensor[0]

        return tensor
