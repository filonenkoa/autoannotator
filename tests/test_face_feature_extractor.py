import numpy as np

from autoannotator.feature_extraction.faces.models.model_adaface import (
    ConfigAdaface,
    FaceFeatureExtractorAdaface,
)
from autoannotator.utils.image_alignment import ImageAlignmentSimilarityTransform
from autoannotator.feature_extraction.faces.models.ensemble import (
    FaceFeatureExtractionEnsemle,
)
from autoannotator.feature_extraction.faces.models.model_insightface import (
    FaceFeatureExtractorInsightface,
)
from autoannotator.utils.image_reader import ImageReader


def _aligned_face() -> np.ndarray:
    reader = ImageReader()
    input_img = reader("assets/images/ms_01.jpg")
    keypoints = [
        [340.0, 574.0, 1.0],
        [478.0, 503.0, 1.0],
        [403.0, 610.0, 1.0],
        [409.0, 716.0, 1.0],
        [527.0, 657.0, 1.0],
    ]

    regressor = ImageAlignmentSimilarityTransform()
    return regressor(input_img, keypoints)


def test_model_adaface():
    model_adaface = FaceFeatureExtractorAdaface()
    aligned_img = _aligned_face()

    embedding = model_adaface(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_adaface.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)


def test_model_adaface_batch():
    model_adaface = FaceFeatureExtractorAdaface()
    aligned_img = _aligned_face()

    batch = np.stack([aligned_img, aligned_img])
    embeddings = model_adaface(batch)

    assert embeddings.shape[0] == 2
    ground_truth = np.load("assets/binaries/ms_01_embedding_adaface.npy")
    np.testing.assert_allclose(embeddings[0], ground_truth, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(embeddings[1], ground_truth, rtol=1e-03, atol=1e-05)


def test_model_insightace():
    model_insightface = FaceFeatureExtractorInsightface()
    aligned_img = _aligned_face()

    embedding = model_insightface(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_insightface.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)


def test_model_insightace_batch():
    model_insightface = FaceFeatureExtractorInsightface()
    aligned_img = _aligned_face()

    batch = np.stack([aligned_img, aligned_img])
    embeddings = model_insightface(batch)

    assert embeddings.shape[0] == 2
    ground_truth = np.load("assets/binaries/ms_01_embedding_insightface.npy")
    np.testing.assert_allclose(embeddings[0], ground_truth, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(embeddings[1], ground_truth, rtol=1e-03, atol=1e-05)


def test_feature_extractor_ensemble():
    adaface_config = ConfigAdaface()
    adaface_config.device = "cuda"
    model_adaface = FaceFeatureExtractorAdaface()
    model_insightface = FaceFeatureExtractorInsightface()
    ensemble = FaceFeatureExtractionEnsemle(models=[model_adaface, model_insightface])
    aligned_img = _aligned_face()

    embedding = ensemble(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_ensemble.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)


def test_feature_extractor_ensemble_batch():
    adaface_config = ConfigAdaface()
    adaface_config.device = "cuda"
    model_adaface = FaceFeatureExtractorAdaface()
    model_insightface = FaceFeatureExtractorInsightface()
    ensemble = FaceFeatureExtractionEnsemle(models=[model_adaface, model_insightface])
    aligned_img = _aligned_face()

    batch = np.stack([aligned_img, aligned_img])
    embeddings = ensemble(batch)

    assert embeddings.shape[0] == 2
    ground_truth = np.load("assets/binaries/ms_01_embedding_ensemble.npy")
    np.testing.assert_allclose(embeddings[0], ground_truth, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(embeddings[1], ground_truth, rtol=1e-03, atol=1e-05)
