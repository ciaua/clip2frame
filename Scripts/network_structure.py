#!/usr/bin/env python

import lasagne
from lasagne import layers
import theano.tensor as T
from clip2frame import layers as cl


def build_fcn_gaussian_multiscale(mean_std_list):

    input_var_list = list()

    # A stack of convolution layers for each scale
    for ii, [mean, std] in enumerate(mean_std_list):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))
        input_var_list.append(input_var)

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # early conv layers
        earlyconv_list = list()

        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 128),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name='early_conv.{}_1'.format(ii)
        )
        network = layers.MaxPool2DXLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_1'.format(ii)
        )

        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 1),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name='early_conv.{}_2'.format(ii)
        )
        network = layers.MaxPool2DXLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_2'.format(ii)
        )

        earlyconv_list.append(network)

    network = layers.ConcatLayer(earlyconv_list,
                                 axis=1,
                                 cropping=[None, None, 'lower', None],
                                 name='multiscale_concat')

    # late conv layers
    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='late_conv.{}_1'.format(ii)
    )

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='late_conv.{}_2'.format(ii)
    )

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 188, (1, 1), (1, 1),
        name='late_conv.{}_output'.format(ii)
    )

    # Gaussian scanning
    network = layers.ReshapeLayer(network, ([0], [1], [2]))

    network = cl.GaussianScan1DLayer(
        network,
        filter_size=256, init_std=256/4**2,
        stride=1,
        pad='strictsame',
        name='gaussian_filter')

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)
    # network = layers.ReshapeLayer(network, ([0], -1))

    return network, input_var_list


def build_fcn_fixedgaussian_multiscale(mean_std_list):

    input_var_list = list()

    # A stack of convolution layers for each scale
    for ii, [mean, std] in enumerate(mean_std_list):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))
        input_var_list.append(input_var)

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # early conv layers
        earlyconv_list = list()

        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 128),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name='early_conv.{}_1'.format(ii)
        )
        network = layers.MaxPool2DXLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_1'.format(ii)
        )

        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 1),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name='early_conv.{}_2'.format(ii)
        )
        network = layers.MaxPool2DXLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_2'.format(ii)
        )

        earlyconv_list.append(network)

    network = layers.ConcatLayer(earlyconv_list,
                                 axis=1,
                                 cropping=[None, None, 'lower', None],
                                 name='multiscale_concat')

    # late conv layers
    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='late_conv.{}_1'.format(ii)
    )

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 512, (1, 1), (1, 1),
        name='late_conv.{}_2'.format(ii)
    )

    network = layers.Conv2DLayer(
        layers.dropout(network, p=0.5), 188, (1, 1), (1, 1),
        name='late_conv.{}_output'.format(ii)
    )

    # Gaussian scanning
    network = layers.ReshapeLayer(network, ([0], [1], [2]))

    network = cl.FixedGaussianScan1DLayer(
        network,
        filter_size=256, init_std=256/4**2,
        stride=1,
        pad='strictsame',
        name='fixed_gaussian_filter')

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)
    # network = layers.ReshapeLayer(network, ([0], -1))

    return network, input_var_list


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
