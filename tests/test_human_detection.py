import pytest
import numpy as np

from autoannotator.detection.human import HumanDetEnsemble
from autoannotator.detection.human import UniHCPHuman, InternImageHuman, IterDETR, RTDETR
from autoannotator.detection.utils.test_time_augmentation import (
    TTAColorHistogramEqualization,
    TTAHorizontalFlip,
)
from autoannotator.utils.image_reader import ImageReader


def test_model_unihcp():
    # Pass this test until we re-train the model.
    # Original authors of UniHCP require signing an agreement before using their weights.
    return True
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = UniHCPHuman()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([450, 241, 617, 669])
    
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_model_iterdetr():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = IterDETR()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([448, 241, 616, 671])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_model_rtdetr():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = RTDETR()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([449, 240, 616, 670])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_human_detection_ensemble():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    # TODO: Restore this test when we re-train UniHCP
    # models = [UniHCPHuman(), IterDETR(), RTDETR()]
    # hd_ensemble = HumanDetEnsemble(models=models, model_weights=[0.87, 0.941, 0.87], match_iou_thr=0.5)
    
    models = [IterDETR(), RTDETR()]
    hd_ensemble = HumanDetEnsemble(models=models, model_weights=[0.941, 0.87], match_iou_thr=0.5)

    img = reader(img_file)
    detections, meta, _ = hd_ensemble(img)

    expected_bbox = np.array([448, 240, 616, 670])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_human_detection_ensemble_tta_batch():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    models = [IterDETR(), RTDETR()]
    tta_transforms = [TTAColorHistogramEqualization(), TTAHorizontalFlip()]
    hd_ensemble = HumanDetEnsemble(
        models=models,
        model_weights=[0.941, 0.87],
        match_iou_thr=0.5,
        tta=tta_transforms,
    )

    img = reader(img_file)
    batch = np.stack([img, img])

    detections, meta, raw_results = hd_ensemble(batch)

    assert isinstance(detections, list)
    assert len(detections) == 2
    assert all(len(det_list) > 0 for det_list in detections)

    expected_bbox = np.array([448, 240, 616, 670])
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0][0].bbox).astype(np.int32)
    )
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[1][0].bbox).astype(np.int32)
    )

    assert len(meta) == 2
    assert len(raw_results) == 2
    assert all(isinstance(result_dict, dict) for result_dict in raw_results)

    tta_names = {transform.name for transform in tta_transforms}
    for result_dict in raw_results:
        assert isinstance(result_dict, dict)
        assert any(
            any(tta_name in key for tta_name in tta_names) for key in result_dict
        )
