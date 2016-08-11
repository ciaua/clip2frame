import os
from clip2frame import utils, measure
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_all_layers
from lasagne import layers


if __name__ == '__main__':
    # Test Options
    frame_layer_type = 'gaussian'  # 'pre-gaussian'
    test_measure_type_list = ['precision_micro', 'recall_micro', 'f1_micro']

    # test_measure_type_list = ['precision_micro', 'precision_macro',
    #                           'recall_micro', 'recall_macro',
    #                           'f1_micro', 'f1_macro']

    upscale_method = 'naive'  # 'patching'
    threshold_source = 'MedleyDB'  # 'MagnaTagATune'

    # Files and directories
    base_tr_dir = '../data/data.magnatagatune'
    base_te_dir = '../data/data.medleydb'

    model_dir = '../data/models'
    param_fp = '../data/models/model.20160309_111546.npz'
    standardizer_dir = '../data/standardizers'

    tag_tr_fp = '../data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = '../data/data.medleydb/instrument_list.top9.txt'
    tag_conv_fp = \
        '../data/data.medleydb/instrument_list.medleydb_magnatagatune.top9.csv'

    # Default setting
    scale_list = ['logmelspec10000.16000_{}_512_128.0.raw'.format(win_size)
                  for win_size in [512, 1024, 2048, 4096, 8192, 16384]]
    n_scales = len(scale_list)

    feat_dir_list = [os.path.join(base_te_dir, 'sample_features', scale)
                     for scale in scale_list]
    anno_dir = os.path.join(base_te_dir, 'sample_annotations.top9')

    fn_list = os.listdir(anno_dir)

    # Load tag list
    tag_idx_list = utils.get_test_tag_indices(tag_tr_fp, tag_te_fp, tag_conv_fp)

    # Standardizer dir
    std_fp_list = [os.path.join(standardizer_dir, 'scaler.{}.pkl'.format(
        scale.replace('.raw', '')))
        for scale in scale_list]
    std_list = [utils.unpickle(std_fp) for std_fp in std_fp_list]

    # Network options
    network_type = 'fcn_gaussian_multiscale'
    n_early_conv = 2
    early_pool_size = 4
    network_options = {
        'early_conv_dict_list': [
            {'conv_filter_list': [(32, 8) for ii in range(n_early_conv)],
             'pool_filter_list': [early_pool_size
                                  for ii in range(n_early_conv)],
             'pool_stride_list': [None for ii in range(n_early_conv)]}
            for ii in range(n_scales)
        ],
        'late_conv_dict': {
            'conv_filter_list': [(512, 1), (512, 1)],
            'pool_filter_list': [None, None],
            'pool_stride_list': [None, None]
        },
        'dense_filter_size': 1,
        'scan_dict': {
            'scan_filter_list': [256],
            'scan_std_list': [256/early_pool_size**n_early_conv],
            'scan_stride_list': [1],
        },
        'final_pool_function': T.mean,  # T.max
        'input_size_list': [128 for nn in range(n_scales)],
        'output_size': 188,
        'p_dropout': 0.5
    }
    network, input_var_list, pr_func = utils.make_network_multiscale_test(
        network_type, n_scales, network_options
    )

    pool_filter_list = \
        network_options['early_conv_dict_list'][0]['pool_filter_list']
    upscale_factor = np.prod(pool_filter_list)

    # Load params
    utils.load_model(param_fp, network)
    if frame_layer_type == 'gaussian':
        idx_layer = -3
    elif frame_layer_type == 'pre-gaussian':
        idx_layer = -4

    # Get frame output layer
    layer_list = get_all_layers(network)
    frame_output_layer = layer_list[idx_layer]

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
