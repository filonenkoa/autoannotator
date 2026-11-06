import numpy as np
from typing import List, Tuple

from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import ImageColorFormat
from autoannotator.types.custom_typing import Tuple2i, Tuple2f, Tuple3i, Tuple3f
from autoannotator.types.faces import Face
from autoannotator.utils.image_preprocessing import resize_image, np2onnx, normalize_image
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root

__all__ = ['YOLOv7DetectionConfig', 'YOLOv7']


_ROOT = get_project_root()


class YOLOv7DetectionConfig(DetectionConfig):
    """ YOLOv7w6-Face object detector config """
    weights: str = f'{_ROOT}/weights/detection/faces/yolov7-w6-face.onnx'
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/yolov7-w6-face.onnx"
    conf_thresh: float = 0.4
    nms_thresh: float = 0.5
    input_size: Tuple2i = (640, 640)
    mean: Tuple3i | Tuple3f = (0., 0., 0.)
    std: Tuple3i | Tuple3f = (1.0, 1.0, 1.0)


class YOLOv7(BaseDetector):
    """
    YOLOv7Face onnx interface. Refer to: https://github.com/derronqi/yolov7-face.
    The onnx model includes nms and detect layer. Supported models: YOLOv7w6

    WiderFace metrics: easy 96.4, medium 95.0, hard 88.3 (mean: 0.9373)

    Arguments:
        config (DetectionConfig): detector config
    """
    def __init__(self, config: DetectionConfig = YOLOv7DetectionConfig()):
        super(YOLOv7, self).__init__(config)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.config.input_size

    @property
    def name(self):
        return 'YOLOv7w6-Face'

    def _predict(self, img: np.ndarray) -> List[Face]:
        x, shift, scale = self._preprocess(img)
        raw_out = self._forward(x)
        out = self._postprocess(raw_out, shift, scale)
        return out

    def predict_batch(self, imgs: List[np.ndarray]) -> List[List[Face]]:
        if not imgs:
            return []

        # If the exported ONNX model has a fixed batch size of 1, don't attempt batched inference.
        try:
            in_shape = self.session.get_inputs()[0].shape
            if (
                len(imgs) > 1
                and isinstance(in_shape, (list, tuple))
                and len(in_shape) >= 1
                and isinstance(in_shape[0], int)
                and in_shape[0] == 1
            ):
                return [self._predict(img) for img in imgs]
        except Exception:  # noqa: BLE001
            pass

        try:
            resized_imgs: list[np.ndarray] = []
            shifts: list[Tuple2i] = []
            scales: list[Tuple2f] = []
            for img in imgs:
                img_r, shift, scale = resize_image(
                    img, size=self.input_size, keep_ratio=True, position="center"
                )
                img_r = normalize_image(img_r, mean=self.config.mean, std=self.config.std)
                resized_imgs.append(img_r)
                shifts.append(shift)
                scales.append(scale)

            blob = np2onnx(resized_imgs, color_mode=ImageColorFormat.RGB)
            raw_out = self.session.run([], {self.input_name: blob})

            det = raw_out[0]
            if det.ndim == 2:
                det = det[None, ...]

            batch_size = int(det.shape[0])
            out: list[list[Face]] = []
            for b in range(batch_size):
                out.append(self._postprocess([det[b]], shift=shifts[b], scale=scales[b]))
            return out
        except Exception:  # noqa: BLE001
            return [self._predict(img) for img in imgs]

    def _forward(self, blob):
        net_outs = self.session.run([], {self.input_name: blob})
        return net_outs

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple2i, Tuple2f]:
        img, shift, scale = resize_image(img, size=self.input_size, keep_ratio=True, position='center')

        img = normalize_image(img, mean=self.config.mean, std=self.config.std)

        img = np2onnx(img, color_mode=ImageColorFormat.RGB)

        return img, shift, scale

    def _postprocess(self, raw_out, shift=(0, 0), scale=(1.0, 1.0)) -> List[Face]:
        res = []
        raw_out = raw_out[0]

        for item in raw_out:
            bbs, score, cls_id, kps = item[0:4], item[4], item[5], item[6:]

            bbs[[0, 2]] = (bbs[[0, 2]] - shift[0]) / scale[0]
            bbs[[1, 3]] = (bbs[[1, 3]] - shift[1]) / scale[1]

            kps = kps.reshape((-1, 3))

            kps[:, 0] = (kps[:, 0] - shift[0]) / scale[0]
            kps[:, 1] = (kps[:, 1] - shift[1]) / scale[1]

            if score > self.config.conf_thresh:
                res.append(Face(
                    cls_id=0,
                    score=score,
                    bbox=bbs.tolist(),  # xyxy
                    landmarks=kps.tolist()  # 5 x [x, y, conf]
                ))
        return res
