from typing import List, Dict, Union
import numpy as np
import numpy.typing as npt

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess, \
    tensorleap_unlabeled_preprocess, tensorleap_input_encoder, tensorleap_gt_encoder, \
    tensorleap_metadata, tensorleap_custom_visualizer, tensorleap_custom_metric, tensorleap_custom_loss
from keras.losses import CategoricalCrossentropy

from cifar10_resnet.utils import metadata_animal, metadata_fly, metadata_label_name, metadata_gt_label
from cifar10_resnet.data.preprocess import preprocess_func
from cifar10_resnet.config import CONFIG


# Preprocess Function
@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    train_X, train_Y, test_X, test_Y, train_idxs, _ = preprocess_func()
    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    train = PreprocessResponse(data={'images': train_X, 'labels': train_Y, 'subset_name': 'train'}, sample_ids=train_idxs.tolist(), sample_id_type=int)
    # Validation have to be added, but Tensorleap should not see validation when testing labeling to prevent data leakage.
    # So, add validation with few samples.
    val = PreprocessResponse(data={'images': test_X, 'labels': test_Y, 'subset_name': 'val'}, sample_ids=list(range(20)), sample_id_type=int)
    response = [train, val]
    return response


@tensorleap_unlabeled_preprocess()
def unlabeled_data() -> PreprocessResponse:
    train_X, train_Y, test_X, test_Y, _, unlabeled_idxs = preprocess_func()
    if len(unlabeled_idxs) > 0:
        return PreprocessResponse(data={'images': train_X, 'subset_name': 'unlabeled'}, sample_ids=unlabeled_idxs.tolist(), sample_id_type=int)
    else:
        return None

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image = preprocess.data['images'][idx].astype('float32')
    return image

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
        return preprocess.data['labels'][idx].astype('float32')

@tensorleap_custom_loss('ce')
def ce_loss(gt: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # Model outputs logits, hence from_logits=True
    return CategoricalCrossentropy(from_logits=True, reduction='none')(gt, y_pred).numpy()

@tensorleap_custom_metric('predicted_label')
def get_predicted_label(pred: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    # argmax is invariant under (log_)softmax, so take it directly on the logits
    bs = pred.shape[0]
    pred_index = pred.argmax(axis=-1)
    return pred_index.reshape(bs)

@tensorleap_custom_metric('accuracy', direction=MetricDirection.Upward)
def get_accuracy(pred: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    pred_index = pred.argmax(axis=-1)
    target_index = ground_truth.argmax(axis=-1)
    acc = (target_index == pred_index).astype(float)
    return acc

@tensorleap_metadata('sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx

@tensorleap_metadata('metadata')
def metadata_dict(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    if preprocess.data['subset_name'] == 'unlabeled':
        gt_label = -1
        label_name = 'unlabeled'
        fly = 'unlabeled'
        animal = 'unlabeled'
    else:
        one_hot_digit = gt_encoder(idx, preprocess)
        digit = one_hot_digit.argmax()  # Returns the indices of the maximum values along an axis
        digit_int = int(digit)
        gt_label = metadata_gt_label(digit_int)
        label_name = metadata_label_name(digit_int)
        fly = metadata_fly(digit_int)
        animal = metadata_animal(digit_int)

    res = {
        "gt_label": gt_label,
        "gt_label_name": label_name,
        "fly": fly,
        "animal": animal
    }
    return res


@tensorleap_custom_visualizer('horizontal_bar_lm', LeapDataType.HorizontalBar)
def horizontal_bar_visualizer_with_labels_name(data: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    data = np.squeeze(data)
    labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(data.shape[-1])]
    return LeapHorizontalBar(data, labels_names)


if __name__ == '__main__':
    leap_binder.check()
