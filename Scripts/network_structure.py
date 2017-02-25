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

        # standardization
        network = layers.standardize(
            network,
            mean, std,
            shared_axes=[0, 1, 2]
        )

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

        # standardization
        network = layers.standardize(
            network,
            mean, std,
            shared_axes=[0, 1, 2]
        )

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


def build_fcn_multiscale(mean_std_list):

    input_var_list = list()

    # A stack of convolution layers for each scale
    for ii, [mean, std] in enumerate(mean_std_list):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))
        input_var_list.append(input_var)

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # standardization
        network = layers.standardize(
            network,
            mean, std,
            shared_axes=[0, 1, 2]
        )

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

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)
    # network = layers.ReshapeLayer(network, ([0], -1))

    return network, input_var_list
