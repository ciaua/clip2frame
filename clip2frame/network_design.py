#!/usr/bin/env python

import lasagne
from lasagne import layers
import theano.tensor as T
from clip2frame import layers as cl


# Parts
def conv_layers(network, conv_dict, total_stride,
                init_input_size=1, p_dropout=0,
                W_list=None, b_list=None, return_params=False,
                base_name=''):
    '''
    conv_dict:
        a dictionary containing the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

        conv_filter_list: list
            each element is in the form (# of filters, filter length)

        pool_filter_list: list
            each element is an integer

        pool_stride_list: list
            each element is int or None

    '''

    conv_filter_list = conv_dict['conv_filter_list']
    pool_filter_list = conv_dict['pool_filter_list']
    pool_stride_list = conv_dict['pool_stride_list']
    assert(len(conv_filter_list) ==
           len(pool_filter_list) ==
           len(pool_stride_list))
    n_layers = len(conv_filter_list)

    out_W_list = []
    out_b_list = []

    # shared variables
    if type(W_list) is list:
        if len(W_list) != n_layers:
            assert(False)
    elif W_list is None:
        W_list = [lasagne.init.GlorotUniform() for kk in range(n_layers)]
    else:
        assert(False)

    if type(b_list) is list:
        if len(b_list) != n_layers:
            assert(False)
    elif b_list is None:
        b_list = [lasagne.init.Constant(0.) for kk in range(n_layers)]
    else:
        assert(False)

    for ii, [conv_filter, pool_filter, pool_stride, W, b] in enumerate(
            zip(conv_filter_list, pool_filter_list, pool_stride_list,
                W_list, b_list)):
        if len(conv_filter) == 2:
            n_filters, filter_len = conv_filter
            conv_stride = 1
            pad = 'strictsamex'
        elif len(conv_filter) == 3:
            n_filters, filter_len, conv_stride = conv_filter
            if conv_stride is None:
                conv_stride = 1

            if conv_stride == 1:
                pad = 'strictsamex'
            else:
                pad = 'valid'
        total_stride *= conv_stride

        if ii == 0:
            feat_dim = init_input_size
        else:
            feat_dim = 1

        network = cl.Conv2DXLayer(
            lasagne.layers.dropout(network, p=p_dropout),
            num_filters=n_filters, filter_size=(filter_len, feat_dim),
            stride=(conv_stride, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad=pad,
            W=W, b=b,
            name='{}.conv{}'.format(base_name, ii)
        )
        out_W_list.append(network.W)
        out_b_list.append(network.b)

        if (pool_filter is not None) or (pool_filter > 1):
            if pool_stride is None:
                stride = None
                total_stride *= pool_filter
            else:
                stride = (pool_stride, 1)
                total_stride *= pool_stride

            # network = lasagne.layers.MaxPool2DLayer(
            network = cl.MaxPool2DXLayer(
                network,
                pool_size=(pool_filter, 1),
                stride=stride,
                ignore_border=False,
                # pad='strictsamex',
                name='{}.maxpool{}'.format(base_name, ii)
            )

    if return_params:
        return network, total_stride, out_W_list, out_b_list
    else:
        return network, total_stride


# Multi-label (4D input)
def fcn_gaussian_multiscale(
        input_var_list,
        early_conv_dict_list,
        late_conv_dict,
        dense_filter_size,
        scan_dict,
        final_pool_function=T.max,
        input_size_list=[128], output_size=188,
        p_dropout=0.5
        ):
    '''
    early_conv_dict_list: list
        each element in the list is a dictionary containing the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer

    pool_filter_list: list
        each element is an integer

    pool_stride_list: list
        each element is int or None

    '''
    assert(len(early_conv_dict_list) == len(input_var_list) ==
           len(input_size_list))

    # early conv layers
    conv_network_list = list()
    total_stride_list = list()
    for jj, [early_conv_dict, input_var, input_size] in enumerate(zip(
            early_conv_dict_list, input_var_list, input_size_list)):
        input_network = lasagne.layers.InputLayer(
            shape=(None, 1, None, input_size), input_var=input_var)

        total_stride = 1
        network, total_stride = conv_layers(input_network, early_conv_dict,
                                            total_stride,
                                            init_input_size=input_size,
                                            p_dropout=0,
                                            base_name='early{}'.format(jj))
        total_stride_list.append(total_stride)
        conv_network_list.append(network)

    '''
    # upsampling
    conv_network_list = [cl.LocalExtend(net, axis=2, extend_size=ts)
                         for net, ts in zip(conv_network_list,
                                            total_stride_list)]
    '''
    network = layers.ConcatLayer(conv_network_list,
                                 axis=1,
                                 cropping=[None, None, 'lower', None],
                                 name='MultisourceConcatenate')

    # late conv layers (dense layers)
    network, total_stride = conv_layers(network, late_conv_dict,
                                        total_stride,
                                        init_input_size=1,
                                        p_dropout=p_dropout,
                                        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # Gaussian scanning
    scan_filter_list = scan_dict['scan_filter_list']
    scan_stride_list = scan_dict['scan_stride_list']
    scan_std_list = scan_dict['scan_std_list']
    network = layers.ReshapeLayer(network, ([0], [1], -1))

    for ii, [filter_size, stride_size, init_std] in enumerate(
            zip(scan_filter_list, scan_stride_list, scan_std_list)):
        network = cl.GaussianScan1DLayer(network,
                                         filter_size, init_std,
                                         stride=stride_size,
                                         pad='strictsame',
                                         # pad='full',
                                         name='scan_{}'.format(ii))

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network


def fcn_fixedgaussian_multiscale(
        input_var_list,
        early_conv_dict_list,
        late_conv_dict,
        dense_filter_size,
        scan_dict,
        final_pool_function=T.max,
        input_size_list=[128], output_size=188,
        p_dropout=0.5
        ):
    '''
    early_conv_dict_list: list
        each element in the list is a dictionary containing the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer

    pool_filter_list: list
        each element is an integer

    pool_stride_list: list
        each element is int or None

    '''
    assert(len(early_conv_dict_list) == len(input_var_list) ==
           len(input_size_list))

    # early conv layers
    conv_network_list = list()
    total_stride_list = list()
    for jj, [early_conv_dict, input_var, input_size] in enumerate(zip(
            early_conv_dict_list, input_var_list, input_size_list)):
        input_network = lasagne.layers.InputLayer(
            shape=(None, 1, None, input_size), input_var=input_var)

        total_stride = 1
        network, total_stride = conv_layers(input_network, early_conv_dict,
                                            total_stride,
                                            init_input_size=input_size,
                                            p_dropout=0,
                                            base_name='early{}'.format(jj))
        total_stride_list.append(total_stride)
        conv_network_list.append(network)

    '''
    # upsampling
    conv_network_list = [cl.LocalExtend(net, axis=2, extend_size=ts)
                         for net, ts in zip(conv_network_list,
                                            total_stride_list)]
    '''
    network = layers.ConcatLayer(conv_network_list,
                                 axis=1,
                                 cropping=[None, None, 'lower', None],
                                 name='MultisourceConcatenate')

    # late conv layers (dense layers)
    network, total_stride = conv_layers(network, late_conv_dict,
                                        total_stride,
                                        init_input_size=1,
                                        p_dropout=p_dropout,
                                        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # Gaussian scanning
    scan_filter_list = scan_dict['scan_filter_list']
    scan_stride_list = scan_dict['scan_stride_list']
    scan_std_list = scan_dict['scan_std_list']
    network = layers.ReshapeLayer(network, ([0], [1], -1))

    for ii, [filter_size, stride_size, init_std] in enumerate(
            zip(scan_filter_list, scan_stride_list, scan_std_list)):
        network = cl.FixedGaussianScan1DLayer(
            network,
            filter_size, init_std,
            stride=stride_size,
            pad='strictsame',
            # pad='full',
            name='scan_{}'.format(ii)
        )

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network


def fcn_multiscale(
        input_var_list,
        early_conv_dict_list,
        late_conv_dict,
        dense_filter_size,
        final_pool_function=T.max,
        input_size_list=[128], output_size=188,
        p_dropout=0.5
        ):
    '''
    early_conv_dict_list: list
        each element in the list is a dictionary containing the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    late_conv_dict: dict
        it contains the following keys:
        'conv_filter_list', 'pool_filter_list', 'pool_stride_list'

    dense_filter_size: int
        the filter size of the final dense-like conv layer

    pool_filter_list: list
        each element is an integer

    pool_stride_list: list
        each element is int or None

    '''
    assert(len(early_conv_dict_list) == len(input_var_list) ==
           len(input_size_list))

    # early conv layers
    conv_network_list = list()
    total_stride_list = list()
    for jj, [early_conv_dict, input_var, input_size] in enumerate(zip(
            early_conv_dict_list, input_var_list, input_size_list)):
        input_network = lasagne.layers.InputLayer(
            shape=(None, 1, None, input_size), input_var=input_var)

        total_stride = 1
        network, total_stride = conv_layers(input_network, early_conv_dict,
                                            total_stride,
                                            init_input_size=input_size,
                                            p_dropout=0,
                                            base_name='early{}'.format(jj))
        total_stride_list.append(total_stride)
        conv_network_list.append(network)

    # Concatenate
    network = layers.ConcatLayer(conv_network_list,
                                 axis=1,
                                 cropping=[None, None, 'lower', None],
                                 name='MultisourceConcatenate')

    # late conv layers (dense layers)
    network, total_stride = conv_layers(network, late_conv_dict,
                                        total_stride,
                                        init_input_size=1,
                                        p_dropout=p_dropout,
                                        base_name='late')

    # frame output layer. every frame has a value
    network = cl.Conv2DXLayer(
        lasagne.layers.dropout(network, p=p_dropout),
        num_filters=output_size, filter_size=(dense_filter_size, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform()
    )

    # pool
    network = layers.GlobalPoolLayer(network,
                                     pool_function=final_pool_function)
    network = layers.ReshapeLayer(network, ([0], -1))

    return network
