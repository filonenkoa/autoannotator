import numpy as np
from typing import List, Dict, Tuple

from autoannotator.detection.utils.wbf import weighted_boxes_fusion
from autoannotator.types.faces import Face
from autoannotator.types.base import Detection
from autoannotator.detection.core.base_detector import BaseDetector


class FaceDetEnsemble:
    """
    This is a face detection ensemble class

    Arguments:
       models (List[BaseDetector]): list of face detectors
    """

    def __init__(
        self,
        models: List[BaseDetector],
        match_iou_thr: float = 0.5,
        model_weights: List[float] = None,
    ):
        """
        Constructor

        Arguments:
            models (List[BaseDetector]): All the models that should be used in the inference.
            match_iou_thr (float): IoU threshold to match Detections, default 0.5
            model_weights (List[float]): model weights that are used to merge predictions into one
        """
        super(FaceDetEnsemble, self).__init__()
        self.models = models
        self.model_weights = model_weights
        self.match_iou_thr = match_iou_thr

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[List[Face], List[dict], Dict[str, List[Detection]]]:
        """
        Run inference with the ensemble of models on a given image

        Arguments:
            img (np.ndarray): The input image.

        Returns:
            predictions (List[Face]): List of detected faces
            meta (List[dict]): List of matching meta data
            results (Dict[str, List[Face]]): dict of per model predictions
        """

        results = {}
        for model in self.models:
            res = model(img)
            results[str(model.name)] = res

        predictions, meta = self.reduce(results)

        return predictions, meta, results

    def predict_batch(
        self, imgs: List[np.ndarray]
    ) -> Tuple[List[List[Face]], List[List[dict]], Dict[str, List[List[Detection]]]]:
        """Run inference on a batch of images.

        Returns per-image reduced predictions + per-model raw results.
        """
        results: Dict[str, List[List[Detection]]] = {}
        for model in self.models:
            results[str(model.name)] = model.predict_batch(imgs)

        predictions_batch: list[list[Face]] = []
        meta_batch: list[list[dict]] = []
        for i in range(len(imgs)):
            per_img = {k: v[i] for k, v in results.items()}
            preds, meta = self.reduce(per_img)
            predictions_batch.append(preds)
            meta_batch.append(meta)

        return predictions_batch, meta_batch, results

    def reduce(
        self, results: Dict[str, List[Detection]]
    ) -> Tuple[List[Face], List[dict]]:
        """
        Reduces ensemble models predictions into single prediction with Weighted Boxes Fusion Algorithm

        Arguments:
            results (Dict[List[Face]]): Dict of predicted faces {'model1': faces_arr1, 'model2: faces_arr2, ...}
        Returns:
            (List[Face]): Reduced List of detected faces
        """
        boxes_list = []
        scores_list = []
        kp_list = []
        labels_list = []

        for key, model_preds in results.items():
            boxes = []
            scores = []
            kps = []
            labels = []
            for face in model_preds:
                boxes.append(face.bbox)
                scores.append(face.score)
                kps.append(np.array(face.landmarks).flatten().tolist())
                labels.append(face.cls_id)

            boxes_list.append(boxes)
            scores_list.append(scores)
            kp_list.append(kps)
            labels_list.append(labels)

        annotations = {
            "labels": labels_list,
            "scores": scores_list,
            "boxes": boxes_list,
            "kps": kp_list,
        }
        w_annotations = weighted_boxes_fusion(
            annotations, weights=self.model_weights, iou_thr=self.match_iou_thr
        )

        out = []
        meta = []
        for ann in w_annotations:
            out.append(
                Face(
                    cls_id=ann["label"],
                    score=ann["score"],
                    bbox=ann["bbox"].tolist(),
                    landmarks=ann["kps"].reshape(-1, 3).tolist(),
                )
            )
            meta.append(ann["meta"])
        return out, meta
