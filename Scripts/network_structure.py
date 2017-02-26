#!/usr/bin/env python

import lasagne
from lasagne import layers
import theano.tensor as T
from clip2frame import layers as cl


def build_fcn_gaussian_multiscale(num_scales):

    input_var_list = list()
    earlyconv_list = list()

    # A stack of convolution layers for each scale
    for ii in range(num_scales):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # early conv layers
        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 128),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='strictsamex',
            name='early_conv.{}_1'.format(ii)
        )
        network = layers.MaxPool2DLayer(
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
            pad='strictsamex',
            name='early_conv.{}_2'.format(ii)
        )
        network = layers.MaxPool2DLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_2'.format(ii)
        )

        input_var_list.append(input_var)
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
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='late_conv.{}_output'.format(ii)
    )

    # Gaussian scanning
    network = layers.ReshapeLayer(network, ([0], [1], [2]))
    nogaussian_layer = network

    network = cl.GaussianScan1DLayer(
        network,
        filter_size=256, init_std=256/4**2,
        stride=1,
        pad='strictsame',
        name='gaussian_filter')
    gaussian_layer = network

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)

    return network, input_var_list, nogaussian_layer, gaussian_layer


def build_fcn_fixedgaussian_multiscale(num_scales):

    input_var_list = list()
    earlyconv_list = list()

    # A stack of convolution layers for each scale
    for ii in range(num_scales):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # early conv layers
        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 128),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='strictsamex',
            name='early_conv.{}_1'.format(ii)
        )
        network = layers.MaxPool2DLayer(
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
            pad='strictsamex',
            name='early_conv.{}_2'.format(ii)
        )
        network = layers.MaxPool2DLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_2'.format(ii)
        )

        input_var_list.append(input_var)
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
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='late_conv.{}_output'.format(ii)
    )

    # Gaussian scanning
    network = layers.ReshapeLayer(network, ([0], [1], [2]))
    nogaussian_layer = network

    network = cl.FixedGaussianScan1DLayer(
        network,
        filter_size=256, init_std=256/4**2,
        stride=1,
        pad='strictsame',
        name='fixed_gaussian_filter')
    gaussian_layer = network

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)

    return network, input_var_list, nogaussian_layer, gaussian_layer


def build_fcn_multiscale(num_scales):

    input_var_list = list()
    earlyconv_list = list()

    # A stack of convolution layers for each scale
    for ii in range(num_scales):
        # input tensor
        input_var = T.tensor4('input.{}'.format(ii))

        # input layer
        network = lasagne.layers.InputLayer(
            shape=(None, 1, None, 128), input_var=input_var)

        # early conv layers
        network = cl.Conv2DXLayer(
            network,
            num_filters=32, filter_size=(8, 128),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad='strictsamex',
            name='early_conv.{}_1'.format(ii)
        )
        network = layers.MaxPool2DLayer(
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
            pad='strictsamex',
            name='early_conv.{}_2'.format(ii)
        )
        network = layers.MaxPool2DLayer(
            network,
            pool_size=(4, 1),
            stride=(4, 1),
            ignore_border=False,
            name='early_maxpool.{}_2'.format(ii)
        )

        input_var_list.append(input_var)
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
        nonlinearity=lasagne.nonlinearities.sigmoid,
        name='late_conv.{}_output'.format(ii)
    )
    nogaussian_layer = network

    # pool
    network = layers.GlobalPoolLayer(network, pool_function=T.mean)

    gaussian_layer = None
    return network, input_var_list, nogaussian_layer, gaussian_layer
