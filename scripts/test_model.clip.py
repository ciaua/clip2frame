import theano.tensor as T
import numpy as np
from clip2frame import utils, measure


if __name__ == '__main__':
    # Options
    scale_list = [
        "scale0",
        "scale1",
        "scale2",
    ]

    test_measure_type_list = ['mean_auc_y', 'mean_auc_x', 'map_y', 'map_x']
    n_top_tags_te = 50  # 188

    # Files
    param_fp = 'models/sample_model.npz'
    tag_tr_fp = 'data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = 'data.magnatagatune/tag_list.top{}.txt'.format(n_top_tags_te)
    data_dir = 'data.magnatagatune/sample_exp_data'

    # Load tag list
    tag_tr_list = utils.read_lines(tag_tr_fp)
    tag_te_list = utils.read_lines(tag_te_fp)

    label_idx_list = [tag_tr_list.index(tag) for tag in tag_te_list]

    # Load data
    X_te_list, y_te = utils.load_data_multiscale_test(data_dir, scale_list)
    n_sources = len(scale_list)

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
            for ii in range(n_sources)
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
        'input_size_list': [128 for nn in range(n_sources)],
        'output_size': 188,
        'p_dropout': 0.5
    }

    network, input_var, pr_func = \
        utils.make_network_multiscale_test(
            network_type, n_sources, network_options
        )

    # Load params
    utils.load_model(param_fp, network)

    # Predict
    pred_list_raw = utils.predict_multiscale(X_te_list, pr_func)
    pred_all_raw = np.vstack(pred_list_raw)

    pred_all = pred_all_raw[:, label_idx_list]
    anno_all = y_te[:, label_idx_list]

    for measure_type in test_measure_type_list:
        measure_func = getattr(measure, measure_type)
        score = measure_func(anno_all, pred_all)
        print("{}:\t\t{:.4f}".format(measure_type, score))
