import os
import tensorflow as tf
from leap_binder import (preprocess_func_leap, input_encoder_leap, gt_encoder, metadata_dict, metadata_sample_index
, horizontal_bar_visualizer_with_labels_name, ce_loss)
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from cifar10_resnet.config import CONFIG

prediction_type1 = PredictionTypeHandler('classes', CONFIG['LABELS_NAMES'])

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/resnet.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return cnn

@tensorleap_integration_test()
def check_custom_intgeration(idx, subset):
    plot_vis = True
    inpt = input_encoder_leap(idx, subset)
    resnet = load_model()
    y_pred = resnet(inpt)
    sample_index = metadata_sample_index(idx, subset)
    dict_metadata = metadata_dict(idx, subset)

    horizontal_bar_pred = horizontal_bar_visualizer_with_labels_name(y_pred)

    if plot_vis:
        visualize(horizontal_bar_pred, 'Prediction')

    gt = gt_encoder(idx, subset)
    ls = ce_loss(gt, y_pred)

    horizontal_bar_gt = horizontal_bar_visualizer_with_labels_name(gt)

    if plot_vis:
        visualize(horizontal_bar_gt, 'GT')


def check_custom_integration():

    responses = preprocess_func_leap()

    check_custom_intgeration(0, responses[0])
    print("Custom tests finished successfully")

if __name__ == '__main__':
    check_custom_integration()



