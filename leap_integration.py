"""Tensorleap integration for the CIFAR-10 ResNet classifier.

Decorator-style integration:
- preprocess reads CIFAR-10 from the Tensorleap data volume (config-driven path),
  caching a keras download into the volume on a cache-miss.
- the input encoder reproduces the training-time transform (resize 32->224, /255).
- the model is a Keras ResNet whose 10-way output already has softmax applied
  (outputs are probabilities, not logits).
"""

import os
from typing import List, Dict

import numpy as np
import yaml

from code_loader.contract.datasetclasses import (
    PreprocessResponse,
    PredictionTypeHandler,
    SamplePreprocessResponse,
)
from code_loader.contract.enums import (
    DataStateType,
    LeapDataType,
    MetricDirection,
    DatasetMetadataType,
)
from code_loader.contract.visualizer_classes import LeapImage, LeapHorizontalBar
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_load_model,
    tensorleap_custom_loss,
    tensorleap_metadata,
    tensorleap_custom_visualizer,
    tensorleap_custom_metric,
    tensorleap_integration_test,
)

# --------------------------------------------------------------------------- #
# Config (config-driven, never hardcoded in logic)
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_ROOT, "cifar10_resnet", "project_config.yaml")

with open(_CONFIG_PATH, "r") as _f:
    CONFIG = yaml.safe_load(_f)

LABELS: List[str] = list(CONFIG["LABELS_NAMES"])
NUM_CLASSES = len(LABELS)
DATA_DIR = os.environ.get("CIFAR10_DATA_DIR", CONFIG["DATA_VOLUME_DIR"])
SUBSET = {
    DataStateType.training: int(CONFIG["SUBSET_TRAIN"]),
    DataStateType.validation: int(CONFIG["SUBSET_VAL"]),
}
MODEL_PATH = os.path.join(_ROOT, "model", "resnet.h5")


# --------------------------------------------------------------------------- #
# Data loading (from the data volume; download+cache on a cache-miss)
# --------------------------------------------------------------------------- #
def _load_cifar_arrays():
    """Return (X uint8 (N,32,32,3), y int (N,)) from the data volume, caching a
    keras download into the volume the first time only."""
    x_path = os.path.join(DATA_DIR, "cifar10_x.npy")
    y_path = os.path.join(DATA_DIR, "cifar10_y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        os.makedirs(DATA_DIR, exist_ok=True)
        from keras.datasets import cifar10
        (data_x, data_y), _ = cifar10.load_data()
        np.save(x_path, data_x.astype(np.uint8))
        np.save(y_path, np.squeeze(data_y).astype(np.int64))
    return np.load(x_path), np.load(y_path)


# --------------------------------------------------------------------------- #
# Preprocess
# --------------------------------------------------------------------------- #
@tensorleap_preprocess()
def preprocess() -> List[PreprocessResponse]:
    from sklearn.model_selection import train_test_split

    data_x, data_y = _load_cifar_arrays()
    indices = np.arange(len(data_x))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    responses: List[PreprocessResponse] = []
    for state, split_idx in (
        (DataStateType.training, train_idx),
        (DataStateType.validation, val_idx),
    ):
        sel = split_idx[: SUBSET[state]]
        sample_ids = [str(int(g)) for g in sel]
        data: Dict[str, object] = {
            "images": data_x[sel],
            "labels": data_y[sel],
            "id_to_pos": {sid: pos for pos, sid in enumerate(sample_ids)},
        }
        responses.append(
            PreprocessResponse(sample_ids=sample_ids, data=data, state=state)
        )
    return responses


# --------------------------------------------------------------------------- #
# Input encoder — reproduces the training transform: resize 32->224, scale to [0,1]
# --------------------------------------------------------------------------- #
@tensorleap_input_encoder(name="image", channel_dim=-1)
def image_input(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    from scipy.ndimage import zoom

    pos = preprocess.data["id_to_pos"][sample_id]
    image = preprocess.data["images"][pos]  # (32,32,3) uint8
    resized = zoom(image, (7, 7, 1)) / 255.0  # -> (224,224,3) in [0,1]
    return resized.astype(np.float32)


# --------------------------------------------------------------------------- #
# Model — Keras ResNet; single output of 10 softmax probabilities
# --------------------------------------------------------------------------- #
PREDICTION_TYPES = [
    PredictionTypeHandler(name="classes", labels=LABELS, channel_dim=-1)
]


@tensorleap_load_model(PREDICTION_TYPES)
def load_model():
    import keras

    return keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------------------------------- #
# Ground-truth encoder — one-hot over the 10 classes
# --------------------------------------------------------------------------- #
@tensorleap_gt_encoder(name="classes")
def class_gt(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    pos = preprocess.data["id_to_pos"][sample_id]
    label = int(preprocess.data["labels"][pos])
    one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot


# --------------------------------------------------------------------------- #
# Custom loss — per-sample categorical cross-entropy (model outputs probabilities)
# --------------------------------------------------------------------------- #
@tensorleap_custom_loss("categorical_crossentropy")
def cce_loss(prediction: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.clip(np.asarray(prediction, dtype=np.float32), 1e-7, 1.0)
    y = np.asarray(gt, dtype=np.float32)
    return (-np.sum(y * np.log(pred), axis=-1)).astype(np.float32)  # (B,)


# --------------------------------------------------------------------------- #
# Metrics — per-sample, batch-aligned 1D
# --------------------------------------------------------------------------- #
@tensorleap_custom_metric("accuracy", direction=MetricDirection.Upward)
def accuracy(prediction: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    y = np.asarray(gt, dtype=np.float32)
    correct = np.argmax(pred, axis=-1) == np.argmax(y, axis=-1)
    return correct.astype(np.float32)  # (B,)


@tensorleap_custom_metric("confidence", direction=MetricDirection.Upward)
def confidence(prediction: np.ndarray) -> np.ndarray:
    """Max softmax probability — an unsupervised confidence/calibration signal."""
    pred = np.asarray(prediction, dtype=np.float32)
    return np.max(pred, axis=-1).astype(np.float32)  # (B,)


# --------------------------------------------------------------------------- #
# Metadata — per-sample descriptors for slicing/analysis (surface as sample_<key>)
# --------------------------------------------------------------------------- #
@tensorleap_metadata(
    "sample",
    {
        "label": DatasetMetadataType.string,
        "label_index": DatasetMetadataType.int,
        "brightness": DatasetMetadataType.float,
    },
)
def meta_sample(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, object]:
    pos = preprocess.data["id_to_pos"][sample_id]
    label_index = int(preprocess.data["labels"][pos])
    image = preprocess.data["images"][pos]
    return {
        "label": LABELS[label_index],
        "label_index": label_index,
        "brightness": float(np.mean(image) / 255.0),
    }


# --------------------------------------------------------------------------- #
# Visualizers
# --------------------------------------------------------------------------- #
@tensorleap_custom_visualizer("input_image", LeapDataType.Image)
def input_image_visualizer(image: np.ndarray) -> LeapImage:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 4:  # batched inside the integration test
        img = img[0]
    disp = (img * 255.0).clip(0, 255).astype(np.uint8)  # platform renders 0-255
    return LeapImage(disp)


@tensorleap_custom_visualizer("class_probabilities", LeapDataType.HorizontalBar)
def class_bar_visualizer(
    prediction: np.ndarray, gt: np.ndarray
) -> LeapHorizontalBar:
    pred = np.asarray(prediction, dtype=np.float32)
    y = np.asarray(gt, dtype=np.float32)
    if pred.ndim == 2:  # batched inside the integration test
        pred = pred[0]
    if y.ndim == 2:
        y = y[0]
    return LeapHorizontalBar(body=pred, labels=LABELS, gt=y)


# --------------------------------------------------------------------------- #
# Integration test — thin: only decorated calls + minimal inference
# --------------------------------------------------------------------------- #
@tensorleap_integration_test()
def integration_test(sample_id: str, preprocess: PreprocessResponse):
    image = image_input(sample_id, preprocess)
    gt = class_gt(sample_id, preprocess)
    model = load_model()
    prediction = model(image)
    _ = cce_loss(prediction, gt)
    _ = accuracy(prediction, gt)
    _ = confidence(prediction)
    _ = meta_sample(sample_id, preprocess)
    _ = input_image_visualizer(image)
    _ = class_bar_visualizer(prediction, gt)


if __name__ == "__main__":
    subsets = preprocess()
    print([(s.state, len(s.sample_ids), s.sample_ids[:3]) for s in subsets])
    for subset in subsets:
        if subset.state not in {DataStateType.training, DataStateType.validation}:
            continue
        for sample_id in subset.sample_ids[:3]:
            integration_test(sample_id, subset)
        print(f"integration_test OK on 3 {subset.state} samples")
