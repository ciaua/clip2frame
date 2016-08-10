import lasagne
import theano.tensor as T
from clip2frame import utils


if __name__ == '__main__':
    # Data options
    data_dir = 'data.magnatagatune/sample_exp_data'
    scale_list = [
        "scale0",
        "scale1",
        "scale2",
    ]

    n_sources = len(scale_list)

    # Training options
    lr = 0.01
    loss_function = lasagne.objectives.binary_crossentropy
    n_epochs = 10
    batch_size = 1  # we use 10 for real data

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

    # Loading data
    print("Loading data...")
    X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te = \
        utils.load_data_multiscale(
            data_dir, scale_list
        )

    network, input_var, lr_var, train_func, val_func, pr_func = \
        utils.make_network_multiscale(
            network_type, loss_function, lr, n_sources,
            network_options, make_pr_func=True
        )

    # Training
    utils.train_multiscale(
        X_tr_list, y_tr, X_va_list, y_va,
        network,
        train_func, val_func,
        n_epochs, batch_size, lr_var
    )
