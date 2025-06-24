from typing import List, Dict, Union
import torch
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
from code_loader.contract.datasetclasses import PreprocessResponse

from cifar10_resnet.utils import metadata_animal, metadata_fly, metadata_label_name, metadata_gt_label
from cifar10_resnet.data.preprocess import preprocess_func
from cifar10_resnet.config import CONFIG


# Preprocess Function
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


def unlabeled_data() -> PreprocessResponse:
    train_X, train_Y, test_X, test_Y, _, unlabeled_idxs = preprocess_func()
    if len(unlabeled_idxs) > 0:
        return PreprocessResponse(data={'images': train_X, 'subset_name': 'unlabeled'}, sample_ids=unlabeled_idxs.tolist(), sample_id_type=int)
    else:
        return None

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
def input_encoder_leap(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    image = preprocess.data['images'][idx].astype('float32')
    return image

# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
        return preprocess.data['labels'][idx].astype('float32')

def get_predicted_label(pred: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    bs = pred.shape[0]
    pred =  F.log_softmax(torch.tensor(pred), dim=1).detach().numpy()
    pred_index = pred.argmax(axis=-1)
    return pred_index.reshape(bs)

def get_accuracy(pred: npt.NDArray[np.float32], ground_truth: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    bs = pred.shape[0]
    pred =  F.log_softmax(torch.tensor(pred), dim=1).detach().numpy()
    pred_index = pred.argmax(axis=-1)
    target_index = ground_truth.argmax(axis=-1)
    acc = (target_index==pred_index).astype(float)
    return acc

def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx

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


def horizontal_bar_visualizer_with_labels_name(data: npt.NDArray[np.float32]) -> LeapHorizontalBar:
    data = np.squeeze(data)
    labels_names = [CONFIG['LABELS_NAMES'][index] for index in range(data.shape[-1])]
    return LeapHorizontalBar(data, labels_names)


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func_leap)
leap_binder.set_unlabeled_data_preprocess(function=unlabeled_data)
leap_binder.set_input(function=input_encoder_leap, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='classes')
leap_binder.set_metadata(function=metadata_sample_index, name='sample_index')
leap_binder.set_metadata(function=metadata_dict, name='metadata')
leap_binder.add_custom_metric(function=get_predicted_label, name="predicted_label")
leap_binder.add_custom_metric(function=get_accuracy, name="accuracy")
leap_binder.set_visualizer(horizontal_bar_visualizer_with_labels_name, 'horizontal_bar_lm', LeapDataType.HorizontalBar)
leap_binder.add_prediction(name='classes', labels=CONFIG['LABELS_NAMES'])

if __name__ == '__main__':
    leap_binder.check()