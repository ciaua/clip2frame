import theano
import theano.tensor as T
from lasagne import layers
import numpy as np
from clip2frame import utils, measure
import network_structure as ns


if __name__ == '__main__':
    # Test settings
    build_func = ns.build_fcn_gaussian_multiscale
    test_measure_type_list = ['mean_auc_y', 'mean_auc_x', 'map_y', 'map_x']
    n_top_tags_te = 50  # 188

    # Files
    param_fp = '../data/models/sample_model.npz'
    standardizer_dir = '../data/standardizers/'
    tag_tr_fp = '../data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = '../data/data.magnatagatune/tag_list.top{}.txt'.format(
        n_top_tags_te)
    data_dir = '../data/data.magnatagatune/sample_exp_data'

    # Model
    scale_list = [
        "logmelspec10000.16000_512_512_128.0.standard",
        "logmelspec10000.16000_1024_512_128.0.standard",
        "logmelspec10000.16000_2048_512_128.0.standard",
        "logmelspec10000.16000_4096_512_128.0.standard",
        "logmelspec10000.16000_8192_512_128.0.standard",
        "logmelspec10000.16000_16384_512_128.0.standard",
    ]

    # Load tag list
    tag_tr_list = utils.read_lines(tag_tr_fp)
    tag_te_list = utils.read_lines(tag_te_fp)

    label_idx_list = [tag_tr_list.index(tag) for tag in tag_te_list]

    # Load data
    X_te_list, y_te = utils.load_data_multiscale_te(data_dir, scale_list)
    n_sources = len(scale_list)

    # Building Network
    print("Building network...")
    num_scales = len(scale_list)
    network, input_var_list, _, _ = build_func(num_scales)

    # Computing loss
    target_var = T.matrix('targets')
    epsilon = np.float32(1e-6)
    one = np.float32(1)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)

    func_pr = theano.function(input_var_list, output_va_var)

    # Load params
    utils.load_model(param_fp, network)

    # Predict
    pred_list_raw = utils.predict_multiscale(X_te_list, func_pr)
    pred_all_raw = np.vstack(pred_list_raw)

    pred_all = pred_all_raw[:, label_idx_list]
    anno_all = y_te[:, label_idx_list]

    for measure_type in test_measure_type_list:
        measure_func = getattr(measure, measure_type)
        score = measure_func(anno_all, pred_all)
        print("{}:\t\t{:.4f}".format(measure_type, score))
