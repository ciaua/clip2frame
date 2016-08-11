#!/usr/bin/env python

import os
import sys
import csv
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import f1_score
from multiprocessing import Pool

from clip2frame import network_design


theano.config.exception_verbosity = 'high'

floatX = theano.config.floatX
epsilon = np.float32(1e-6)
one = np.float32(1)
pf = np.float32(0.5)


# IO
ver = sys.version_info
if ver >= (3, 0):
    import pickle as pk
    opts_write = {'encoding': 'utf-8', 'newline': ''}
    opts_read = {'encoding': 'utf-8'}
else:
    import cPickle as pk
    opts_write = {}
    opts_read = {}


# IO
def read_lines(file_path):
    with open(file_path, 'r') as opdrf:
        data = [term.strip() for term in opdrf.readlines()]
        return data


def read_csv(file_path):
    with open(file_path, 'r', **opts_read) as opdrf:
        csv_reader = csv.reader(opdrf)
        data = [term for term in csv_reader]
        return data


def pickle(file_path, obj, protocol=2):
    """
    For python 3 compatibility, use protocol 2
    """
    if not file_path.endswith('.pkl'):
        file_path += '.pkl'
    with open(file_path, 'wb') as opdwf:
        pk.dump(obj, opdwf, protocol=protocol)


def unpickle(file_path):
    with open(file_path, 'rb') as opdrf:
        data = pk.load(opdrf)
        return data


# Load data
def load_data_multiscale(data_dir, scale_list):
    X_tr_list = list()
    y_tr_list = list()

    X_te_list = list()
    y_te_list = list()

    X_va_list = list()
    y_va_list = list()

    for ii, scale in enumerate(scale_list):
        feat_tr_fp = os.path.join(data_dir,
                                  'feat.tr.{}.npy'.format(scale))
        target_tr_fp = os.path.join(data_dir,
                                    'target.tr.{}.npy'.format(scale))
        feat_va_fp = os.path.join(data_dir,
                                  'feat.va.{}.npy'.format(scale))
        target_va_fp = os.path.join(data_dir,
                                    'target.va.{}.npy'.format(scale))
        feat_te_fp = os.path.join(data_dir,
                                  'feat.te.{}.npy'.format(scale))
        target_te_fp = os.path.join(data_dir,
                                    'target.te.{}.npy'.format(scale))

        X_tr = np.load(feat_tr_fp)
        y_tr = np.load(target_tr_fp)

        X_va = np.load(feat_va_fp)
        y_va = np.load(target_va_fp)

        X_te = np.load(feat_te_fp)
        y_te = np.load(target_te_fp)

        # append
        X_tr_list.append(X_tr)
        y_tr_list.append(y_tr)

        X_te_list.append(X_te)
        y_te_list.append(y_te)

        X_va_list.append(X_va)
        y_va_list.append(y_va)

    y_tr = y_tr_list[0]
    y_va = y_va_list[0]
    y_te = y_te_list[0]

    return X_tr_list, y_tr, X_va_list, y_va, X_te_list, y_te


def load_data_multiscale_test(data_dir, scale_list):
    X_te_list = list()
    y_te_list = list()

    for ii, scale in enumerate(scale_list):
        feat_te_fp = os.path.join(data_dir,
                                  'feat.te.{}.npy'.format(scale))
        target_te_fp = os.path.join(data_dir,
                                    'target.te.{}.npy'.format(scale))

        X_te = np.load(feat_te_fp)
        y_te = np.load(target_te_fp)

        # append
        X_te_list.append(X_te)
        y_te_list.append(y_te)

    y_te = y_te_list[0]

    return X_te_list, y_te


def load_data_multiscale_va(data_dir, scale_list):
    X_va_list = list()
    y_va_list = list()

    for ii, scale in enumerate(scale_list):
        feat_va_fp = os.path.join(data_dir,
                                  'feat.va.{}.npy'.format(scale))
        target_va_fp = os.path.join(data_dir,
                                    'target.va.{}.npy'.format(scale))

        X_va = np.load(feat_va_fp)
        y_va = np.load(target_va_fp)

        # append
        X_va_list.append(X_va)
        y_va_list.append(y_va)

    y_va = y_va_list[0]

    return X_va_list, y_va


# Make networks
def make_network_multiscale(
        network_type, loss_function, lr, n_scales, net_options,
        do_clip=True, make_pr_func=False):
    target_var = T.matrix('targets')
    lr_var = theano.shared(np.array(lr, dtype=floatX))

    print("Building model and compiling functions...")
    if n_scales >= 1:
        input_var_list = [T.tensor4('inputs{}'.format(i))
                          for i in range(n_scales)]
        network = getattr(network_design, network_type)(input_var_list,
                                                        **net_options)
    else:
        # if the network requires input_var not a list, set n_sources=-1
        input_var_list = [T.addbroadcast(T.tensor4('inputs{}'.format(i)))
                          for i in range(1)]
        network = getattr(network_design, network_type)(input_var_list[0],
                                                        **net_options)

    # Compute loss
    prediction = lasagne.layers.get_output(network)
    if do_clip:
        prediction = T.clip(prediction, epsilon, one-epsilon)
    loss = loss_function(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(
        loss, params, learning_rate=lr_var)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if do_clip:
        test_prediction = T.clip(test_prediction, epsilon, one-epsilon)
    test_loss = loss_function(test_prediction, target_var)
    test_loss = test_loss.mean()

    train_func = theano.function(input_var_list+[target_var],
                                 loss, updates=updates)
    val_func = theano.function(input_var_list+[target_var],
                               [test_prediction, test_loss])

    if make_pr_func:
        pr_func = theano.function(input_var_list, test_prediction)
        return network, input_var_list, lr_var, train_func, val_func, pr_func
    else:
        return network, input_var_list, lr_var, train_func, val_func


def make_network_multiscale_test(
        network_type, n_scales, net_options, do_clip=True):
    print("Building model and compiling functions...")
    if n_scales >= 1:
        input_var_list = [T.tensor4('inputs{}'.format(i))
                          for i in range(n_scales)]
        network = getattr(network_design, network_type)(input_var_list,
                                                        **net_options)
    else:
        # if the network requires input_var not a list, set n_sources=-1
        input_var_list = [T.addbroadcast(T.tensor4('inputs{}'.format(i)))
                          for i in range(1)]
        network = getattr(network_design, network_type)(input_var_list[0],
                                                        **net_options)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    if do_clip:
        test_prediction = T.clip(test_prediction, epsilon, one-epsilon)

    pr_func = theano.function(input_var_list, test_prediction)
    return network, input_var_list, pr_func


# Iterate inputs
def iterate_minibatches_multiscale(inputs_list, targets,
                                   batchsize, shuffle=False):
    if type(targets) == np.ndarray:
        n = len(targets)
        k = targets.shape[-1]
        for inputs in inputs_list:
            assert len(inputs) == n

    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [inputs[excerpt] for inputs in inputs_list], \
            targets[excerpt].reshape((-1, k))


def iterate_minibatches_multiscale_feat(inputs_list, batchsize, shuffle=False):
    n = len(inputs_list[0])
    for inputs in inputs_list:
        assert len(inputs) == n
    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [inputs[excerpt] for inputs in inputs_list]


# Functions used in train for recording and printing
def check_best_loss(best_val_loss, val_loss):
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_val_updated = True
    else:
        best_val_updated = False
    return best_val_loss, best_val_updated


def print_in_train(epoch, n_epochs,
                   mean_tr_loss, mean_va_loss,
                   best_va_epoch, best_va_loss):
    print("Epoch {} of {}.".format(epoch, n_epochs))
    print("  training loss:        {:.6f}".format(mean_tr_loss))
    print("  validation loss:      {:.6f}".format(mean_va_loss))
    print("  best va (epoch, loss):({}, {:.6f})".format(
        best_va_epoch, best_va_loss
    ))
    print(" ")


# Multiple input sources
def train_multiscale(
        X_tr_list, y_tr, X_va_list, y_va,
        network,
        train_func, va_func,
        n_epochs, batch_size, lr_var, param_fp=None):

    print("Starting training...")

    best_va_epoch = 0
    best_va_loss = np.inf
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        train_batches = 0

        # Training
        for batch_ in iterate_minibatches_multiscale(X_tr_list, y_tr,
                                                     batch_size,
                                                     shuffle=True):
            inputs_list, targets = batch_
            temp = inputs_list+[targets]
            train_loss_one = train_func(*temp)

            train_loss += train_loss_one
            train_batches += 1
        mean_tr_loss = train_loss/train_batches

        # Validation
        pre_list, mean_va_loss = validate_multiscale(X_va_list, y_va,
                                                     va_func)

        # Check best loss
        best_va_loss, best_va_updated = check_best_loss(
            best_va_loss, mean_va_loss)
        if best_va_updated:
            best_va_epoch = epoch
            if param_fp is not None:
                save_model(param_fp, network)

        # Print the results for this epoch:
        print_in_train(epoch, n_epochs,
                       mean_tr_loss, mean_va_loss,
                       best_va_epoch, best_va_loss)


def validate_multiscale(X_list, y, val_func):
    val_loss = 0
    val_batches = 0
    pre_list = []
    for batch in iterate_minibatches_multiscale(X_list, y, 1, shuffle=False):
        inputs_list, targets = batch
        temp = inputs_list+[targets]
        pre, loss = val_func(*temp)
        val_loss += loss
        val_batches += 1
        pre_list.append(pre)

    mean_val_loss = val_loss / val_batches
    return pre_list, mean_val_loss


def predict_multiscale(X_list, pr_func):
    pre_list = []
    for inputs_list in iterate_minibatches_multiscale_feat(
            X_list, 1, shuffle=False):
        pre = pr_func(*inputs_list)
        pre_list.append(pre)

    return pre_list


# Save/load
def save_model(fp, network):
    np.savez(fp, *lasagne.layers.get_all_param_values(network))


def load_model(fp, network):
    with np.load(fp) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


# Get thresholds
def f1_one(y_target, y_predicted):
    '''
    y_target, y_predicted:
        1D binary array
    '''
    return f1_score(y_target, y_predicted, average='binary')


def f1(Y_target, Y_predicted):
    '''
    Y_target, Y_predicted:
        n x k 2D binary array, where n is the number of data and
        k is the number of tags

    '''
    scores = [f1_one(y_target, y_predicted)
              for y_target, y_predicted in zip(Y_target.T, Y_predicted.T)]
    scores = np.array(scores)
    return scores


def get_measure(arg):
    threshold, prediction, target, step_size, lower_b, measure_func = arg
    pred_binary = ((prediction-threshold) > 0).astype(int)

    measures = measure_func(target, pred_binary)
    return measures


def get_thresholds(pred, target, search_range, step_size, measure_func=f1,
                   n_processes=20):
    '''
    pred: np.array
        prediction from a model
        n x k 2D array, where n is the number of data and
        k is the number of tags

    target: np.array
        groundtruth
        n x k 2D binary array, where n is the number of data and
        k is the number of tags

    search_range: tuple
        the range for searching the thresholds
        (a, b), where a is the lower bound and b is the upper bound

    step_size: float
        searching the threholds in (a, a+step_size, a+2step_size, ..., ...)

    measure_func: function or str
        function defined in the begining of this fild
    '''
    lower_b, upper_b = search_range
    assert(upper_b > lower_b)
    if measure_func == 'f1':
        measure_func = f1

    n_tags = target.shape[1]

    diff = upper_b-lower_b
    n_steps = int(np.floor(diff/step_size))

    threshold_list = [lower_b+ii*step_size for ii in range(n_steps+1)]

    arg_list = []
    for th in threshold_list:
        arg_list.append(
            (th, pred, target, step_size, lower_b, measure_func))
    pool = Pool(processes=n_processes)
    all_measures = np.array(pool.map(get_measure, arg_list))
    pool.close()
    # print(all_measures.shape)

    best_idx_list = np.argmax(all_measures, axis=0)

    best_thresholds = lower_b+best_idx_list*step_size
    best_measures = all_measures[best_idx_list, [ii for ii in range(n_tags)]]
    # print(n_tags, len(best_idx_list))

    return best_thresholds, best_measures


# Upscale array
def shift(array_list, shift_size, axis):
    n_axes = len(array_list[0].shape)
    obj = [slice(None, None, None) for ii in range(n_axes)]
    obj[axis] = slice(shift_size, None, 1)
    obj = tuple(obj)

    pad_width = [(0, 0) for ii in range(n_axes)]
    pad_width[axis] = (0, shift_size)

    out_array_list = [np.pad(array[obj], pad_width, 'constant')
                      for array in array_list]
    return out_array_list


def upscale(func, input_list, method='naive', scale_factor=1,
            in_axis=2, out_axis=2):
    '''
    array: numpy.array

    method: str
        'naive' or 'patching'

    scale_factor: int

    '''

    assert(method in ['naive', 'patching'])
    if method == 'naive':
        array = func(*input_list)[0]
        new_array = np.repeat(array, scale_factor, axis=out_axis)
    elif method == 'patching':
        output_list = [func(*shift(input_list, ii, axis=in_axis))[0]
                       for ii in range(scale_factor)]
        output = np.stack(output_list, axis=out_axis+1)

        new_shape = list(output_list[0].shape)
        new_shape[out_axis] = -1
        new_shape = tuple(new_shape)

        new_array = np.reshape(output, new_shape)

    return new_array


# Process tag list
def get_test_tag_indices(tag_tr_fp, tag_te_fp, tag_conv_fp):
    tag_te_list = read_lines(tag_te_fp)
    tag_conv_dict = dict(read_csv(tag_conv_fp))

    tag_tr_list = read_lines(tag_tr_fp)

    tag_idx_list = [tag_tr_list.index(tag_conv_dict[tag])
                    for tag in tag_te_list]
    return tag_idx_list


if __name__ == '__main__':
    x = T.tensor3()
    func = theano.function([x], [2*x])
