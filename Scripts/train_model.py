import numpy as np
import lasagne
from lasagne import layers
import theano
import theano.tensor as T
import network_structure as ns
from clip2frame import utils

floatX = theano.config.floatX

if __name__ == '__main__':
    # Data options
    data_dir = '../data/data.magnatagatune/sample_exp_data'
    param_fp = '../data/models/sample_model.npz'

    # Training options
    build_func = ns.build_fcn_gaussian_multiscale
    lr = 0.01
    loss_function = lasagne.objectives.binary_crossentropy
    n_epochs = 10
    batch_size = 1  # we use 10 for real data
    scale_list = [
        "logmelspec10000.16000_512_512_128.0.standard",
        "logmelspec10000.16000_1024_512_128.0.standard",
        "logmelspec10000.16000_2048_512_128.0.standard",
        "logmelspec10000.16000_4096_512_128.0.standard",
        "logmelspec10000.16000_8192_512_128.0.standard",
        "logmelspec10000.16000_16384_512_128.0.standard",
    ]

    # Loading data
    print("Loading data...")
    X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te = \
        utils.load_data_multiscale(
            data_dir, scale_list
        )

    # Building Network
    print("Building network...")
    num_scales = len(scale_list)
    network, input_var_list, _, _ = build_func(num_scales)

    # Computing loss
    target_var = T.matrix('targets')
    lr_var = theano.shared(np.array(lr, dtype=floatX))
    output_var = layers.get_output(network)
    epsilon = np.float32(1e-6)
    one = np.float32(1)
    output_var = T.clip(output_var, epsilon, one-epsilon)
    loss_var = loss_function(output_var, target_var)
    loss_var = loss_var.mean()

    params = layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(
        loss_var, params, learning_rate=lr_var)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)
    loss_va_var = loss_function(output_va_var, target_var)
    loss_va_var = loss_va_var.mean()

    func_tr = theano.function(
        input_var_list+[target_var], loss_var, updates=updates)
    func_va = theano.function(
        input_var_list+[target_var], [output_va_var, loss_va_var])

    # Training
    utils.train_multiscale(
        X_tr_list, y_tr, X_va_list, y_va,
        network,
        func_tr, func_va,
        n_epochs, batch_size, lr_var,
        param_fp=param_fp
    )
