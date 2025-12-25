import numpy as np
from typing import Any, Dict, List, Sequence, Tuple

from autoannotator.detection.utils.test_time_augmentation import TestTimeAugmentationBase
from autoannotator.detection.utils.wbf import weighted_boxes_fusion
from autoannotator.types.base import Detection
from autoannotator.detection.core.base_detector import BaseDetector


class HumanDetEnsemble(object):
    """
    This is a human detection ensemble class

    Arguments:
       models (List[BaseDetector]): list of human detectors
    """

    def __init__(
        self,
        models: List[BaseDetector],
        match_iou_thr: float = 0.5,
        model_weights: List[float] | None = None,
        tta: List[TestTimeAugmentationBase] | None = None,
    ):
        """
        Constructor

        Arguments:
            models (List[BaseDetector]): All the models that should be used in the inference.
            match_iou_thr (float): IoU threshold to match Detections, default 0.5
            model_weights (List[float]): model weights that are used to merge predictions into one
        """
        super(HumanDetEnsemble, self).__init__()
        if not models:
            raise ValueError("`models` must contain at least one detector instance")

        self.models = models
        self.match_iou_thr = match_iou_thr
        self.base_weights = list(model_weights) if model_weights is not None else None

        if self.base_weights is not None and len(self.base_weights) != len(self.models):
            raise ValueError(
                "Length of `model_weights` must match number of models in the ensemble"
            )

        self.tta = tta or []
        self.use_tta = len(self.tta) > 0
        self.model_weights = self._expand_weights()

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[List[Detection] | List[List[Detection]], List[dict] | List[List[dict]], Dict[str, List[Detection]] | List[Dict[str, List[Detection]]]]:
        """
        Run inference with the ensemble of models on a given image

        Arguments:
            img (np.ndarray): The input image.

        Returns:
            (List[Detection]): List of detected faces
        """

        images, is_batched = self._normalize_inputs(img)

        batch_predictions: List[List[Detection]] = []
        batch_meta: List[List[dict]] = []
        batch_results: List[Dict[str, List[Detection]]] = []

        for image in images:
            predictions, meta, results = self._infer_single(image)
            batch_predictions.append(predictions)
            batch_meta.append(meta)
            batch_results.append(results)

        if not is_batched:
            return batch_predictions[0], batch_meta[0], batch_results[0]

        return batch_predictions, batch_meta, batch_results

    def reduce(self, results: Dict[str, List[Detection]]) -> Tuple[List[Detection], List[dict]]:
        """
        Reduces ensemble models predictions into single prediction with Weighted Boxes Fusion Algorithm

        Arguments:
            results (Dict[List[Detection]]): Dict of predicted faces {'model1': det_arr1, 'model2: det_arr2, ...}
        Returns:
            (List[Detection]): Reduced List of detected objects
        """
        boxes_list, scores_list, labels_list = [], [], []

        for key, model_preds in results.items():
            boxes, scores, labels = [], [], []
            for detection in model_preds:
                boxes.append(detection.bbox)
                scores.append(detection.score)
                labels.append(detection.cls_id)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        annotations = {'labels': labels_list, 'scores': scores_list, 'boxes': boxes_list}
        w_annotations = weighted_boxes_fusion(
            annotations,
            weights=self.model_weights,
            iou_thr=self.match_iou_thr,
        )

        out = []
        meta = []
        for ann in w_annotations:
            out.append(
                Detection(
                    cls_id=ann['label'],
                    score=ann['score'],
                    bbox=ann['bbox'].tolist(),
                )
            )
            meta.append(ann['meta'])
        return out, meta

    def _infer_single(
        self, img: np.ndarray
    ) -> Tuple[List[Detection], List[dict], Dict[str, List[Detection]]]:
        results: Dict[str, List[Detection]] = {}
        for model in self.models:
            res = model(img)
            results[model.name] = res

        if self.use_tta:
            for tta in self.tta:
                augmented_img, metadata = tta.augment(img)
                for model in self.models:
                    res = model(augmented_img)
                    res = tta.rectify(res, metadata)
                    results[f"{model.name}_{tta.name}"] = res

        predictions, meta = self.reduce(results)
        return predictions, meta, results

    def _normalize_inputs(self, img: Any) -> Tuple[List[np.ndarray], bool]:
        if isinstance(img, np.ndarray):
            if img.ndim == 4:
                return [img[i] for i in range(img.shape[0])], True
            return [img], False

        if isinstance(img, Sequence):
            images = [np.asarray(sample) for sample in img]
            if not images:
                raise ValueError("Input batch cannot be empty")
            return images, True

        raise TypeError(
            "Unsupported input type for HumanDetEnsemble; expected ndarray or sequence of ndarrays"
        )

    def _expand_weights(self) -> List[float] | None:
        if self.base_weights is None:
            return None

        repeats = 1 + len(self.tta) if self.use_tta else 1
        return self.base_weights * repeats
