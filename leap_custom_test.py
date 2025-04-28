import os

import tensorflow as tf
import numpy as np
from os.path import exists
import urllib
from keras.losses import CategoricalCrossentropy

from leap_binder import preprocess_func_leap, input_encoder_leap, gt_encoder, metadata_dict, metadata_sample_index, \
    unlabeled_data, horizontal_bar_visualizer_with_labels_name
from leap_binder import leap_binder
from code_loader.helpers import visualize


def check_all_fuctions(responses, resnet, type):
    plot_vis = True
    for i in range(0, 20):
        concat = np.expand_dims(input_encoder_leap(i, responses), axis=0)
        y_pred = resnet([concat])

        sample_index = metadata_sample_index(i, responses)
        dict_metadata = metadata_dict(i, responses)

        horizontal_bar_pred = horizontal_bar_visualizer_with_labels_name(y_pred.numpy())

        if plot_vis:
            visualize(horizontal_bar_pred, 'Prediction')

        if type != "unlabeled":
            gt = np.expand_dims(gt_encoder(i, responses), axis=0)
            y_true = tf.convert_to_tensor(gt)
            ls = CategoricalCrossentropy()(y_true, y_pred).numpy()

            horizontal_bar_gt = horizontal_bar_visualizer_with_labels_name(y_true.numpy())

            if plot_vis:
                visualize(horizontal_bar_gt, 'GT')


def check_custom_integration():
    check_generic = True

    if check_generic:
        leap_binder.check()
    print("started custom tests")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = ('model/resnet.h5')
    resnet = tf.keras.models.load_model(os.path.join(dir_path, model_path))

    responses = preprocess_func_leap()
    unlabeled_responses = unlabeled_data()

    check_all_fuctions(responses[0], resnet, "labeled")
    check_all_fuctions(unlabeled_responses, resnet, "unlabeled")
    print("Custom tests finished successfully")

if __name__ == '__main__':
    check_custom_integration()



