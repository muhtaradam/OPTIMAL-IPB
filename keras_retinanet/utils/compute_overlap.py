# --------------------------------------------------------
# Pure NumPy pengganti compute_overlap.pyx (Cython)
# Ditambahkan karena file .pyd hasil kompilasi Cython bersifat
# spesifik per versi Python (cp37-win_amd64) dan tidak portable
# ke versi Python lain (cp39, cp312, dst).
#
# Catatan: kalau menjalankan di Python 3.7 dengan file
# compute_overlap.cp37-win_amd64.pyd masih ada di folder yang sama,
# Python akan tetap memprioritaskan file .pyd tersebut (lebih cepat)
# dan mengabaikan file .py ini. File ini hanya dipakai sebagai
# fallback otomatis di versi Python lain (3.9, 3.12, dst).
# --------------------------------------------------------

import numpy as np


def compute_overlap(boxes, query_boxes):
    """
    Args
        boxes: (N, 4) ndarray of float
        query_boxes: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    query_boxes = np.asarray(query_boxes, dtype=np.float64)

    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])

    iw = (
        np.minimum(boxes[:, 2][:, None], query_boxes[:, 2][None, :])
        - np.maximum(boxes[:, 0][:, None], query_boxes[:, 0][None, :])
    )
    ih = (
        np.minimum(boxes[:, 3][:, None], query_boxes[:, 3][None, :])
        - np.maximum(boxes[:, 1][:, None], query_boxes[:, 1][None, :])
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    intersection = iw * ih
    ua = box_areas[:, None] + query_areas[None, :] - intersection

    overlaps = np.where(ua > 0, intersection / ua, 0.0)
    return overlaps.astype(np.float64)
