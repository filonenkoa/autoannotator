from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import cv2
import numpy as np
from copy import copy

from autoannotator.types.base import Detection


class TestTimeAugmentationBase(ABC):
    def __init__(self) -> None:
        pass
    
    def __call__(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        It is expected to return the result of the augmentation and metadata used to restore bounding boxes
        """
        return self.augment(img)
    
    @abstractmethod
    def augment(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        raise NotImplementedError
    
    @abstractmethod
    def rectify(self, predictions: List[Detection], metadata: Optional[Dict[str, Any]]):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self):
        """Return detector name"""
        raise NotImplementedError
    

class TTAColorHistogramEqualization(TestTimeAugmentationBase):
    """
    Color histogram normalization
    """
    def __init__(self) -> None:
        super().__init__()
        
    def augment(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        output_img = np.copy(img)
        cv2.normalize(img, output_img, 0, 255, cv2.NORM_MINMAX)
        return output_img, None
    
    def rectify(self, predictions: List[Detection], metadata: Optional[Dict[str, Any]]) -> List[Detection]:
        return predictions
    
    @property
    def name(self):
        return "ColorHistogramEqualization"
    
    
class TTAHorizontalFlip(TestTimeAugmentationBase):
    """
    Minmax color normalization
    """
    def __init__(self) -> None:
        super().__init__()
        
    def augment(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        output_img = cv2.flip(img, 1)
        return output_img, {"original_shape": img.shape}
    
    def rectify(self, predictions: List[Detection], metadata: Optional[Dict[str, Any]]) -> List[Detection]:
        """
        Flips the bounding boxes in place
        """
        if not metadata:
            return predictions

        img_width = metadata["original_shape"][1]
        for prediction in predictions:
            bbox = copy(prediction.bbox)
            bbox[0] = img_width - prediction.bbox[2]
            bbox[2] = img_width - prediction.bbox[0]
            prediction.bbox = bbox
        return predictions
    
    @property
    def name(self):
        return "HorizontalFlip"

