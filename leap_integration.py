import os
import tensorflow as tf
from leap_binder import (preprocess_func_leap, input_encoder_leap, gt_encoder, metadata_dict, metadata_sample_index,
                         horizontal_bar_visualizer_with_labels_name, ce_loss, get_predicted_label, get_accuracy)
from code_loader import leap_binder as binder
from code_loader.plot_functions.visualize import visualize
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from cifar10_resnet.config import CONFIG

prediction_type = PredictionTypeHandler('classes', CONFIG['LABELS_NAMES'])

binder.leap_analysis_configuration.deterministic_results = True

@tensorleap_load_model([prediction_type])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(dir_path, 'model', 'resnet18.h5')
    return tf.keras.models.load_model(model_path)

@tensorleap_integration_test()
def check_integration(idx, subset):
    plot_vis = False

    inpt = input_encoder_leap(idx, subset)
    resnet = load_model()
    y_pred = resnet(inpt)

    sample_index = metadata_sample_index(idx, subset)
    dict_metadata = metadata_dict(idx, subset)
    pred_label = get_predicted_label(y_pred)

    horizontal_bar_pred = horizontal_bar_visualizer_with_labels_name(y_pred)
    if plot_vis:
        visualize(horizontal_bar_pred, 'Prediction')

    if subset.data['subset_name'] != 'unlabeled':
        gt = gt_encoder(idx, subset)
        loss = ce_loss(gt, y_pred)
        acc = get_accuracy(y_pred, gt)

        horizontal_bar_gt = horizontal_bar_visualizer_with_labels_name(gt)
        if plot_vis:
            visualize(horizontal_bar_gt, 'GT')


if __name__ == '__main__':
    responses = preprocess_func_leap()
    check_integration(0, responses[0])
    print("Integration test finished successfully")
