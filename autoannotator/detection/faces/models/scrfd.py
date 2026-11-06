import numpy as np
from typing import List, Tuple

from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import ImageColorFormat
from autoannotator.types.custom_typing import Tuple2f, Tuple2i, Tuple3f, Tuple3i
from autoannotator.types.faces import Face
from autoannotator.utils.image_preprocessing import resize_image, np2onnx, normalize_image
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


__all__ = ['SCRFDDetectionConfig', 'SCRFD']


_ROOT = get_project_root()


class SCRFDDetectionConfig(DetectionConfig):
    """ SCRFD_10G_KPS object detector config """
    weights: str = f'{_ROOT}/weights/detection/faces/scrfd_10g_kps.onnx'
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/scrfd_10g_kps.onnx"
    conf_thresh: float = 0.4
    nms_thresh: float = 0.5
    input_size: Tuple2i = (640, 640)
    mean: Tuple3f | Tuple3i = (0.5, 0.5, 0.5)
    std: Tuple3f | Tuple3i = (0.50196, 0.50196, 0.50196)


class SCRFD(BaseDetector):
    """
    SCRFD onnx interface. Refer to: https://github.com/deepinsight/insightface/tree/master/detection/scrfd.
    Supported models: SCRFD_10G_KPS
    WiderFace metrics: easy 95.40, medium 94.01, hard 82.80 (mean: 0.9073)

    Arguments:
        config (DetectionConfig): detector config
    """

    models_cfg = {
        6: (3, (8, 16, 32), 2, False),
        9: (3, (8, 16, 32), 2, True),
        10: (5, (8, 16, 32, 64, 128), 1, False),
        15: (5, (8, 16, 32, 64, 128), 1, True),
    }

    def __init__(self, config: DetectionConfig = SCRFDDetectionConfig()):
        super(SCRFD, self).__init__(config)

        self.config = config
        self.nms_thresh = config.nms_thresh
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_size = self.config.input_size
        self.fmc, self._strides, self._na, self.use_kps = self.models_cfg[len(outputs)]

        self.center_cache = {}

    @property
    def name(self):
        return 'SCRFD_10G_KPS'

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
            # If shape inspection fails for any reason, proceed with best-effort batching.
            pass

        # Try true batched inference; fall back to per-image if the exported model is fixed-batch.
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
            net_outs = self.session.run(self.output_names, {self.input_name: blob})

            input_height = int(blob.shape[2])
            input_width = int(blob.shape[3])

            is_batched = getattr(net_outs[0], "ndim", 0) == 3
            batch_size = int(blob.shape[0])

            def _forward_one(b: int):
                scores_list = []
                bboxes_list = []
                kpss_list = []
                fmc = self.fmc
                for idx, stride in enumerate(self._strides):
                    if is_batched:
                        scores = net_outs[idx][b]
                        bbox_preds = net_outs[idx + fmc][b] * stride
                        if self.use_kps:
                            kps_preds = net_outs[idx + fmc * 2][b] * stride
                    else:
                        scores = net_outs[idx]
                        bbox_preds = net_outs[idx + fmc] * stride
                        if self.use_kps:
                            kps_preds = net_outs[idx + fmc * 2] * stride

                    height = input_height // stride
                    width = input_width // stride
                    key = (height, width, stride)
                    if key in self.center_cache:
                        anchor_centers = self.center_cache[key]
                    else:
                        anchor_centers = np.stack(
                            np.mgrid[:height, :width][::-1], axis=-1
                        ).astype(np.float32)
                        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                        if self._na > 1:
                            anchor_centers = np.stack([anchor_centers] * self._na, axis=1).reshape(
                                (-1, 2)
                            )
                        if len(self.center_cache) < 100:
                            self.center_cache[key] = anchor_centers

                    pos_inds = np.where(scores >= self.config.conf_thresh)[0]
                    if pos_inds.size == 0:
                        continue
                    bboxes = distance2bbox(anchor_centers, bbox_preds)
                    pos_scores = scores[pos_inds]
                    pos_bboxes = bboxes[pos_inds]
                    scores_list.append(pos_scores)
                    bboxes_list.append(pos_bboxes)
                    if self.use_kps:
                        kpss = distance2kps(anchor_centers, kps_preds)
                        kpss = kpss.reshape((kpss.shape[0], -1, 2))
                        pos_kpss = kpss[pos_inds]
                        kpss_list.append(pos_kpss)
                return scores_list, bboxes_list, kpss_list

            out: list[list[Face]] = []
            for b in range(batch_size):
                raw_out = _forward_one(b)
                out.append(self._postprocess(raw_out, shift=shifts[b], scale=scales[b]))
            return out
        except Exception:  # noqa: BLE001
            return [self._predict(img) for img in imgs]

    def _forward(self, blob):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._strides):
            # If model support batch dim, take first output
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            # If model doesn't support batching take output as is
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._na > 1:
                    anchor_centers = np.stack([anchor_centers] * self._na, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.config.conf_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple2i, Tuple2f]:
        img, shift, scale = resize_image(img, size=self.input_size, keep_ratio=True, position='center')

        img = normalize_image(img, mean=self.config.mean, std=self.config.std)

        img = np2onnx(img, color_mode=ImageColorFormat.RGB)
        return img, shift, scale

    def _postprocess(self, raw_out, shift=(0, 0), scale=(1.0, 1.0)) -> List[Face]:
        scores_list, bboxes_list, kpss_list = raw_out

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list)
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - shift[0]) / scale[0]
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - shift[1]) / scale[1]

        if self.use_kps:
            kpss = np.vstack(kpss_list)
            for i in range(len(kpss)):
                for j in range(len(kpss[i])):
                    kpss[i, j, 0] = (kpss[i, j, 0] - shift[0]) / scale[0]
                    kpss[i, j, 1] = (kpss[i, j, 1] - shift[1]) / scale[1]

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        scores = scores[keep]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        res = []
        for score, bb, kp in zip(scores, det, kpss):
            kp = np.hstack([kp, np.ones((5, 1))]).astype(np.int32)
            res.append(Face(
                cls_id=0,   # todo:
                score=score,
                bbox=bb.tolist()[:-1],        # xyxy
                landmarks=kp.tolist()    # 5 x [x, y, conf]
            ))
        return res

    def _nms(self, dets: np.ndarray) -> np.ndarray:
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
