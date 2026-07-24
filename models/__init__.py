"""Final improved PointNet++ model."""

from .model import Model


def build_model(args, num_classes=2):
    return Model(args, num_classes=num_classes)
