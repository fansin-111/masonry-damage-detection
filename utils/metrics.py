"""Metrics for binary point-wise segmentation."""

import numpy as np


class SegmentationMetrics:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.seen = np.zeros(num_classes, dtype=np.int64)
        self.correct = np.zeros(num_classes, dtype=np.int64)
        self.union = np.zeros(num_classes, dtype=np.int64)

    def update(self, prediction, target):
        prediction = np.asarray(prediction).reshape(-1)
        target = np.asarray(target).reshape(-1)
        for label in range(self.num_classes):
            predicted = prediction == label
            actual = target == label
            self.seen[label] += np.count_nonzero(actual)
            self.correct[label] += np.count_nonzero(predicted & actual)
            self.union[label] += np.count_nonzero(predicted | actual)

    def compute(self):
        iou = np.divide(
            self.correct,
            self.union,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.union != 0,
        )
        class_accuracy = np.divide(
            self.correct,
            self.seen,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=self.seen != 0,
        )
        return {
            "accuracy": float(self.correct.sum() / max(self.seen.sum(), 1)),
            "mean_class_accuracy": float(class_accuracy.mean()),
            "class_iou": iou,
            "miou": float(iou.mean()),
        }
