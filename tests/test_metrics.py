import numpy as np

from lawn_segmentation.metrics import confusion_matrix, summarize_confusion


def test_confusion_matrix_has_true_rows_and_prediction_columns():
    matrix = confusion_matrix(np.array([[0, 2], [1, 1]]), np.array([[0, 1], [2, 1]]), 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 1, 0]]


def test_summary_reports_per_class_iou_and_miou():
    result = summarize_confusion(np.diag([2, 3, 4]), {0: "background", 1: "grass", 2: "soil_grass"})
    assert result["pixel_accuracy"] == 1.0
    assert result["miou"] == 1.0
    assert result["iou"]["grass"] == 1.0
