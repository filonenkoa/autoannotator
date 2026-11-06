# coding: utf-8

import numpy as np
from collections import Counter, defaultdict

_eps = 1e-12


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """

    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if len(boxes_list) == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes, new_box)

    # ious[boxes[0] != new_box] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def cluster_boxes(bboxes, iou_thr):
    # Cluster boxes
    clusters_centers = np.empty((0, 4))
    clusters = []
    centers = np.empty((0, 4))
    for i, bbox in enumerate(bboxes):
        # compute clusters centers
        # cluster_sizes = np.array([len(cluster) for cluster in clusters])[:, np.newaxis]
        # if len(clusters):
        #     centers = clusters_centers / cluster_sizes

        index, best_iou = find_matching_box_fast(centers, bbox, iou_thr)

        if index == -1:
            # generate new cluster
            clusters.append([i])
            centers = np.vstack([centers, bbox.copy()])
            clusters_centers = np.vstack((clusters_centers, bbox.copy()))
        else:
            # extend existing cluster
            clusters[index].append(i)
            clusters_centers[index] += bbox.copy()
    return clusters


def weighted_arr_fusion(arr, scores, model_weights, score_mode='mean'):
    _modes = ['mean', 'max', 'box_and_model_mean']

    weights = (scores * model_weights)[:, np.newaxis]
    weights = weights / (weights.sum() + _eps)              # normalize weights

    arr = arr * weights
    arr = arr.sum(axis=0)

    if score_mode == 'mean':
        score = np.mean(scores)
    elif score_mode == 'box_and_model_mean':
        score = np.mean(scores * model_weights)
    elif score_mode == 'max':
        score = np.max(scores)
    else:
        raise NotImplementedError(f'Unknown score fusion mode {score_mode}. Available mods: {_modes}')

    return arr, score


def select_label(labels, model_weights, mode='weighted'):
    _modes = ['weighted', 'frequency']
    if mode == 'weighted':
        l_scores = defaultdict(list)
        for l, weight in zip(labels, model_weights):
            l_scores[l].append(weight)

        max_score = -1
        out_label = -1
        for label, scores in l_scores.items():
            if np.mean(scores) >= max_score:
                out_label = label
    elif mode == 'frequency':
        out_label = Counter(labels).most_common(1)
    else:
        raise NotImplementedError(f'Unknown label selection mode: {mode}. Available mods: {_modes}')
    return out_label


def preprocess_annotations(annotations):
    # Filter out empty values
    annotations['labels'] = [arr for arr in annotations['labels'] if len(arr) > 0]
    annotations['scores'] = [arr for arr in annotations['scores'] if len(arr) > 0]
    annotations['boxes'] = [arr for arr in annotations['boxes'] if len(arr) > 0]
    kps_arr = annotations.get('kps', None)
    if kps_arr is not None:
        annotations['kps'] = [arr for arr in annotations['kps'] if len(arr) > 0]

    num_models = len(annotations['scores'])
    model_indices = []
    model_indices_old = []
    for i, subarr in enumerate(annotations['scores']):
        model_indices.extend([i for _ in range(len(subarr))])
        model_indices_old.extend([i for i in range(len(subarr))])

    model_indices = np.array(model_indices, dtype=np.int32)
    model_indices_old = np.array(model_indices_old, dtype=np.int32)

    # Important empty-case handling:
    # If all models returned zero predictions, keep dtypes consistent (np arrays)
    # so downstream code doesn't see python lists.
    if len(annotations['labels']) == 0:
        annotations['labels'] = np.empty((0,), dtype=np.int32)
        annotations['scores'] = np.empty((0,), dtype=np.float32)
        annotations['boxes'] = np.empty((0, 4), dtype=np.float32)
        if kps_arr is not None:
            annotations['kps'] = np.empty((0, 0), dtype=np.float32)
        return annotations, model_indices, model_indices_old, num_models

    annotations['labels'] = np.concatenate(annotations['labels'], dtype=np.int32)
    annotations['scores'] = np.concatenate(annotations['scores'], dtype=np.float32)
    annotations['boxes'] = np.concatenate(annotations['boxes'], dtype=np.float32)
    if kps_arr is not None:
        annotations['kps'] = np.concatenate(annotations['kps'], dtype=np.float32)
    return annotations, model_indices, model_indices_old, num_models


def weighted_boxes_fusion(
        annotations: dict,
        weights=None,
        iou_thr=0.55,
        conf_type='mean',
):
    """
    Performs weighted fusion of ensemble predictions.

    :param annotations:
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param conf_type: how to calculate confidence in weighted boxes.
        'mean': average value,
        'max': maximum value,
        'box_and_model_mean': box and model wise hybrid weighted average,

    :return: weighted_annotations: list of dict weighted annotations
    """
    annotations, model_indices, model_indices_old, num_models = preprocess_annotations(annotations)

    # Fast-path for empty predictions.
    if isinstance(annotations.get('scores'), np.ndarray) and annotations['scores'].size == 0:
        return []

    if weights is None:
        weights = np.ones_like(annotations['scores'])
    else:
        assert len(weights) == num_models
        weights = np.array(weights)[model_indices]

    labels_arr = annotations['labels']
    scores_arr = annotations['scores']
    bboxes_arr = annotations['boxes']
    kps_arr = annotations.get('kps', None)

    clusters = cluster_boxes(bboxes_arr, iou_thr)

    weighted_annotations = []
    for c_ind in range(len(clusters)):
        indices = clusters[c_ind]

        labels = labels_arr[indices]
        boxes = bboxes_arr[indices]
        scores = scores_arr[indices]
        model_ids = model_indices[indices]
        model_weights = weights[indices]

        label = select_label(labels, model_weights, mode='weighted')
        wbox, score = weighted_arr_fusion(boxes, scores, model_weights, score_mode=conf_type)

        if kps_arr is not None:
            kps = kps_arr[indices]
            wkps, _ = weighted_arr_fusion(kps, scores, model_weights, score_mode=conf_type)
        else:
            wkps = None

        witem = {
            'label': label,
            'score': score,
            'bbox': wbox,
            'kps': wkps,
            'meta': {
                'unique_models': np.unique(model_ids),
                'clusters': list(zip(model_ids, model_indices_old[indices])),
            }
        }
        weighted_annotations.append(witem)
    return weighted_annotations
