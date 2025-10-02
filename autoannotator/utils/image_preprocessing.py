import cv2
import math
import numpy as np
from typing import Tuple, List, Union

from autoannotator.types.base import ImageColorFormat
from autoannotator.types.custom_typing import Tuple2f, Tuple2i, Tuple3i, Tuple3f


__all__ = ["resize_image", "np2onnx", "normalize_image"]


def _compute_new_shape(
    img0_shape: Tuple2i, img1_shape: Tuple2i, keep_ratio: bool = True
) -> Tuple2i:
    """
    Compute resize HxW dimensions

    Arguments:
        img0_shape (Tuple[int, int]): The input image H0 x W0 shape
        img1_shape (Tuple[int, int]): The target H1 x W1 shape
        keep_ratio (bool): Whether to resize keeping ratio or not
    Returns:
        Tuple[int, int]: New width x New height to resize to
    """
    h0, w0 = img0_shape
    h1, w1 = img1_shape
    if keep_ratio:
        img_ratio = float(w0) / h0

        w_ratio = float(w1) / float(w0)
        h_ratio = float(h1) / float(h0)

        if w_ratio < h_ratio:
            new_w = w1
            new_h = int(new_w / img_ratio)
        else:
            new_h = h1
            new_w = int(new_h * img_ratio)

        return new_h, new_w
    else:
        return h1, w1


def resize_image(
    img: np.ndarray,
    size: Tuple[int, int],
    keep_ratio: bool = True,
    position: str = "center",
    value: int = 0,
) -> Tuple[np.ndarray, Tuple2i, Tuple2f]:
    """
    Resize image

    Arguments:
        img (np.ndarray): The input image H0 x W0 x 3
        size (Tuple[int, int]): The target H1 x W1 size
        keep_ratio (bool): Whether to resize keeping ratio or not
        position (str): Position to insert resized image (`topleft` and `center` options are available).
        value (int): Border value
    Returns:
        (np.ndarray): Resized image (H1 x W1 x 3)
        (Tuple[int, int]): xy translation shift
        (Tuple[float, float]): xy resize scale (resized to original scale)
    """
    h1, w1 = size
    h0, w0 = img.shape[:2]

    new_h, new_w = _compute_new_shape(
        img0_shape=(h0, w0), img1_shape=(h1, w1), keep_ratio=keep_ratio
    )

    if keep_ratio:
        s = float(new_h) / h0
        scale = (s, s)
    else:
        scale = (float(new_w) / w0, float(new_h) / h0)

    resized_img = cv2.resize(
        img, (new_w, new_h)
    )  # dsize in cv2.resize is in (w x h) format

    if position == "center":
        out_img = np.ones((h1, w1, 3), dtype=np.uint8) * value
        x1 = (w1 - new_w) // 2
        y1 = (h1 - new_h) // 2
        x2 = x1 + new_w
        y2 = y1 + new_h
        assert 0 <= x1 <= w1 and 0 <= x2 <= w1
        assert 0 <= y1 <= h1 and 0 <= y2 <= h1
        out_img[y1:y2, x1:x2, :] = resized_img
        shift = (x1, y1)
    elif position == "top_left":
        out_img = np.ones((h1, w1, 3), dtype=np.uint8) * value
        out_img[:new_h, :new_w, :] = resized_img
        shift = (0, 0)
    elif position == "only":
        out_img = resized_img
        shift = (0, 0)
    else:
        raise NotImplementedError(
            f"Only `center`, `top_left`, and `only` positions are supported. Got: {position}"
        )

    return out_img, shift, scale


def np2onnx(
    img: np.ndarray | List[np.ndarray], color_mode: ImageColorFormat = ImageColorFormat.RGB
) -> np.ndarray:
    """
    Convert numpy image to onnx-friendly input format

    Arguments:
        img (np.ndarray): The input image numpy array HxWx3 in rgb format
        color_mode (str): output color mode (rgb or bgr)
    Returns:
        (np.ndarray): ONNX-friendly input (BxCxHxW)
    """
    if isinstance(img, list):
        img = np.asarray(img)

    if color_mode == ImageColorFormat.BGR:
        img = img[..., ::-1]

    img = np.asarray(img, dtype=np.float32)

    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    elif img.ndim != 4:
        raise ValueError("np2onnx expects input with 3 or 4 dimensions")

    img = img.transpose(0, 3, 1, 2)
    return img


def normalize_image(
    img: np.ndarray, mean: Union[Tuple3i, Tuple3f], std: Union[Tuple3i, Tuple3f]
) -> np.ndarray:
    """
    Normalize numpy rgb image

    Arguments:
        img (np.ndarray): The input image numpy array HxWx3 in rgb format
        mean (List[float]): normalized pixels mean list for 3 rgb channels in the range [0, 1]
        std (List[float]): normalized pixels std pixels list for 3 rgb channels in the range [0, 1]
    Returns:
        (np.ndarray): Normalized image
    """
    mean = np.array(mean)
    std = np.array(std)

    if img.dtype == np.uint8 and mean.max() < 1.0:
        img = img / 255
    img = (img - mean) / std
    return img


def tile_image(
        img: np.ndarray,
        smaller_side_parts: int = 2,
        bigger_side_parts: int = 2,
        overlap: float = 0.2
) -> Tuple[List[np.ndarray], List[list]]:
    """
    Splits input image into smaller_side_parts x bigger_side_parts tiles with overlap
    Args:
        img: np.array - image to tile (HxWx3)
        smaller_side_parts: number of tiles along the smallest dimension
        bigger_side_parts: number of tiles along the largest dimension
        overlap: tiles overlap

    Returns:
        images: list of image tiles (np.array, HxWx3)
        windows: list of the corresponding tile coordinates relative to the original image (x_min, y_min, x_max, y_max)
    """
    height, width = img.shape[:2]

    # Assign bigger and smaller sides' partitions
    if height > width:
        h_parts = bigger_side_parts
        w_parts = smaller_side_parts
    elif height == width:
        h_parts = bigger_side_parts
        w_parts = bigger_side_parts
    else:
        h_parts = smaller_side_parts
        w_parts = bigger_side_parts

    h_step = int(height / h_parts)
    w_step = int(width / w_parts)

    # Creating windows for detection with overlap
    windows = []
    images = []
    for h in range(h_parts):
        h_start = h * h_step
        h_end = h_start + h_step
        h_start = math.ceil(h_start * (1 - overlap))
        h_end = math.floor(h_end * (1 + overlap))

        for w in range(w_parts):
            w_start = w * w_step
            w_end = w_start + w_step
            w_start = math.ceil(w_start * (1 - overlap))
            w_end = math.floor(w_end * (1 + overlap))
            x1, y1, x2, y2 = w_start, h_start, min(w_end, width), min(h_end, height)
            windows.append([x1, y1, x2, y2])
            images.append(img[y1:y2, x1:x2, :])

    return images, windows
