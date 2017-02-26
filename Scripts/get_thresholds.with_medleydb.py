from clip2frame import utils
import numpy as np
import theano
from lasagne import layers
import network_structure as ns


if __name__ == '__main__':
    # Search options
    search_range = (0, 1)
    step_size = 1e-4
    measure_type = 'f1'

    # Choosing the function for building the model
    build_func = ns.build_fcn_gaussian_multiscale

    # Files and directories
    data_te_dir = '../data/data.medleydb/sample_exp_data'
    param_fp = '../data/models/sample_model.npz'
    threshold_fp = '../data/models/sample_threshold.medleydb.npy'

    # Tag paths
    tag_tr_fp = '../data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = '../data/data.medleydb/instrument_list.top9.txt'
    tag_conv_fp = \
        '../data/data.medleydb/instrument_list.medleydb_magnatagatune.top9.csv'

    # Default setting
    scale_list = [
        "logmelspec10000.16000_512_512_128.0.standard",
        "logmelspec10000.16000_1024_512_128.0.standard",
        "logmelspec10000.16000_2048_512_128.0.standard",
        "logmelspec10000.16000_4096_512_128.0.standard",
        "logmelspec10000.16000_8192_512_128.0.standard",
        "logmelspec10000.16000_16384_512_128.0.standard",
    ]

    # Load data
    n_scales = len(scale_list)
    X_va_list, y_va = \
        utils.load_data_multiscale_va(
            data_te_dir, scale_list
        )
    X_list = X_va_list
    y = y_va

    # Building Network
    print("Building network...")
    num_scales = len(scale_list)
    network, input_var_list, nogaussian_layer, gaussian_layer = \
        build_func(num_scales)

    # Load params
    utils.load_model(param_fp, network)

    # Get indices of test tags
    tag_idx_list = utils.get_test_tag_indices(tag_tr_fp, tag_te_fp, tag_conv_fp)

    # Make predicting function
    sym_song_prediction = layers.get_output(network, deterministic=True)
    clip_func = theano.function(input_var_list, [sym_song_prediction])

    # Predict
    clip_prediction = np.vstack(
        [term[0] for term in utils.predict_multiscale(X_list, clip_func)])
    clip_prediction = clip_prediction[:, tag_idx_list]

    print('Searching for thresholds...')
    thresholds, measures = utils.get_thresholds(clip_prediction, y,
                                                search_range, step_size,
                                                measure_func=measure_type,
                                                n_processes=20)
    np.save(threshold_fp, thresholds)
