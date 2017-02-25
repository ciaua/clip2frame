import os
from clip2frame import utils, measure
import numpy as np
import theano
import theano.tensor as T
from lasagne import layers
import network_structure as ns


if __name__ == '__main__':
    # Test Options
    build_func = ns.build_fcn_gaussian_multiscale

    frame_layer_type = 'gaussian'  # 'gaussian' or 'no-gaussian'

    test_measure_type_list = ['precision_micro', 'recall_micro', 'f1_micro']

    # test_measure_type_list = ['precision_micro', 'precision_macro',
    #                           'recall_micro', 'recall_macro',
    #                           'f1_micro', 'f1_macro']

    upscale_method = 'naive'  # 'naive' or 'patching'
    threshold_source = 'MedleyDB'  # 'MedleyDB' or 'MagnaTagATune'

    # Files and directories
    base_te_dir = '../data/data.medleydb'

    model_dir = '../data/models'
    param_fp = '../data/models/model.20160309_111546.npz'
    standardizer_dir = '../data/standardizers'

    tag_tr_fp = '../data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = '../data/data.medleydb/instrument_list.top9.txt'
    tag_conv_fp = \
        '../data/data.medleydb/instrument_list.medleydb_magnatagatune.top9.csv'

    # Default setting
    upscale_factor = 16
    scale_list = [
        "logmelspec10000.16000_512_512_128.0.raw",
        "logmelspec10000.16000_1024_512_128.0.raw",
        "logmelspec10000.16000_2048_512_128.0.raw",
        "logmelspec10000.16000_4096_512_128.0.raw",
        "logmelspec10000.16000_8192_512_128.0.raw",
        "logmelspec10000.16000_16384_512_128.0.raw",
    ]
    num_scales = len(scale_list)

    feat_dir_list = [os.path.join(base_te_dir, 'sample_features', scale)
                     for scale in scale_list]
    anno_dir = os.path.join(base_te_dir, 'sample_annotations.top9')

    fn_list = os.listdir(anno_dir)

    # Load tag list
    tag_idx_list = utils.get_test_tag_indices(tag_tr_fp, tag_te_fp, tag_conv_fp)

    # Standardizer dir
    std_fp_list = [os.path.join(standardizer_dir,
                                scale.replace('.raw', ''),
                                'scaler.pkl')
                   for scale in scale_list]
    std_list = [utils.unpickle(std_fp) for std_fp in std_fp_list]

    # Building Network
    print("Building network...")
    num_scales = len(scale_list)
    network, input_var_list, nogaussian_layer, gaussian_layer = \
        build_func(num_scales)

    # Computing loss
    target_var = T.matrix('targets')
    epsilon = np.float32(1e-6)
    one = np.float32(1)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)

    # Load params
    utils.load_model(param_fp, network)

    # Get frame output layer
    if frame_layer_type == 'gaussian':
        frame_output_layer = gaussian_layer
    elif frame_layer_type == 'no-gaussian':
        frame_output_layer = nogaussian_layer

    # Make predicting function
    frame_prediction = layers.get_output(frame_output_layer, deterministic=True)
    func = theano.function(input_var_list, [frame_prediction])

    # label threshold
    if threshold_source in ['magnatagatune', 'MagnaTagATune']:
        thres_fp = os.path.join(
            model_dir, 'threshold.20160309_111546.with_magnatagatune.npy')
        thresholds_raw = np.load(thres_fp)
        thresholds = thresholds_raw[tag_idx_list]
    elif threshold_source in ['MedleyDB', 'medleydb']:
        thres_fp = os.path.join(
            model_dir, 'threshold.20160309_111546.with_medleydb.top9.npy')
        thresholds = np.load(thres_fp)

    # Predict
    anno_all = None
    pred_all = None
    for fn in fn_list:
        anno_fp = os.path.join(anno_dir, fn)
        feat_fp_list = [os.path.join(feat_dir, fn)
                        for feat_dir in feat_dir_list]

        # Process annotation
        anno = np.load(anno_fp)
        n_frames = anno.shape[0]

        feat_list = [np.load(feat_fp).astype('float32')
                     for feat_fp in feat_fp_list]

        # standardize
        feat_list = [standardizer.transform(feat)
                     for feat, standardizer in zip(feat_list, std_list)]

        feat_list = [feat[None, None, :].astype('float32')
                     for feat in feat_list]

        # Predict and upscale
        out_axis = 2
        in_axis = 2
        prediction = utils.upscale(func, feat_list,
                                   upscale_method, upscale_factor,
                                   in_axis, out_axis)
        prediction = prediction[0].T

        prediction = prediction[:n_frames]

        # Narrow down
        prediction = prediction[:, tag_idx_list]

        pred_binary = ((prediction-thresholds) > 0).astype(int)
        try:
            anno_all = np.concatenate([anno_all, anno], axis=0)
            pred_all = np.concatenate([pred_all, pred_binary],
                                      axis=0)
        except:
            anno_all = anno
            pred_all = pred_binary

    test_score_list = list()
    for measure_type in test_measure_type_list:
        measure_func = getattr(measure, measure_type)
        score = measure_func(anno_all, pred_all)
        test_score_list.append(score)
        print("{}:\t\t{:.4f}".format(measure_type, score))
