import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from lasagne import init
from lasagne import nonlinearities
from lasagne import layers
from lasagne.theano_extensions import padding
from lasagne.utils import as_tuple

floatX = theano.config.floatX


# Duplicate of conv1d in lasagne.theano_extensions.conv.
# It is copied just in case it is changed in the future
def conv1d_mc0(input, filters, input_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,)):
    """
    using conv2d with width == 1
    """
    if input_shape is None:
        input_shape_mc0 = None
    else:
        # (b, c, i0) to (b, c, 1, i0)
        input_shape_mc0 = (input_shape[0], input_shape[1], 1, input_shape[2])

    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])

    input_mc0 = input.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)

    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, input_shape=input_shape_mc0,
        filter_shape=filter_shape_mc0, subsample=(1, subsample[0]),
        border_mode=border_mode)
    return conved[:, :, 0, :]  # drop the unused dimension


# modified from lasagne. Add 'strictsamex' for pad.
def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int
        The output size corresponding to the given convolution parameters.

    Raises
    ------
    RuntimeError
        When an invalid padding is specified, a `RuntimeError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif pad == 'strictsamex':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


# modified from lasagne
def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if pad == 'strictsame':
        output_length = input_length
    elif ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


# add 'strictsamex' method for pad
class Pool2DXLayer(layers.Layer):
    """
    2D pooling layer

    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DXLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        if pad == 'strictsamex':
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        if self.pad == 'strictsamex':
            output_shape[2] = pool_output_length(
                input_shape[2],
                pool_size=self.pool_size[0],
                stride=self.stride[0],
                pad='strictsame',
                ignore_border=self.ignore_border,
            )
            output_shape[3] = pool_output_length(
                input_shape[3],
                pool_size=self.pool_size[1],
                stride=self.stride[1],
                pad=0,
                ignore_border=self.ignore_border,
            )
        else:
            output_shape[2] = pool_output_length(
                input_shape[2],
                pool_size=self.pool_size[0],
                stride=self.stride[0],
                pad=self.pad[0],
                ignore_border=self.ignore_border,
            )

            output_shape[3] = pool_output_length(
                input_shape[3],
                pool_size=self.pool_size[1],
                stride=self.stride[1],
                pad=self.pad[1],
                ignore_border=self.ignore_border,
            )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        if self.pad == 'strictsamex':
            assert(self.stride[0] == 1)
            kk = self.pool_size[0]
            ll = int(np.ceil(kk/2.))
            # rr = kk-ll
            # pad = (ll, 0)
            pad = [(ll, 0)]

            length = input.shape[2]

            self.ignore_border = True
            input = padding.pad(input, pad, batch_ndim=2)
            pad = (0, 0)
        else:
            pad = self.pad

        pooled = pool.pool_2d(input,
                              ds=self.pool_size,
                              st=self.stride,
                              ignore_border=self.ignore_border,
                              padding=pad,
                              mode=self.mode,
                              )

        if self.pad == 'strictsamex':
            pooled = pooled[:, :, :length or None, :]

        return pooled


# add 'strictsamex' method for pad
class MaxPool2DXLayer(Pool2DXLayer):
    """
    2D max-pooling layer

    Performs 2D max-pooling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
        super(MaxPool2DXLayer, self).__init__(incoming,
                                              pool_size,
                                              stride,
                                              pad,
                                              ignore_border,
                                              mode='max',
                                              **kwargs)


# add 'strictsamex' method for pad
class Conv2DXLayer(layers.Layer):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'strictsamex'`` pads to the right of the third axis (x axis)
        to keep the same dim as input
        require stride=(1, 1)

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Notes
    -----
    Theano's underlying convolution (:func:`theano.tensor.nnet.conv.conv2d`)
    only supports ``pad=0`` and ``pad='full'``. This layer emulates other modes
    by cropping a full convolution or explicitly padding the input with zeros.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DXLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'strictsamex':
            if not (stride == 1 or stride == (1, 1)):
                raise NotImplementedError(
                    '`strictsamex` padding requires stride=(1, 1) or 1')

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same', 'strictsamex'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        if self.pad == 'strictsamex':
            pad = ('strictsamex', 'valid')
        else:
            pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      # image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            crop_x = self.filter_size[0] // 2
            crop_y = self.filter_size[1] // 2
            conved = conved[:, :, crop_x:-crop_x or None,
                            crop_y:-crop_y or None]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        self.filter_size[0] // 2),
                       (self.filter_size[1] // 2,
                        self.filter_size[1] // 2)]
            elif self.pad == 'strictsamex':
                border_mode = 'valid'
                kk = self.filter_size[0]-1
                rr = kk // 2
                ll = kk-rr
                pad = [(ll, rr),
                       (0, 0)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]

            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      # image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


class GaussianScan1DLayer(layers.Layer):
    """ 1D Adaptive Gaussian filter
    Gaussian filters that scan through the third dimension
    It is implemented with convolution.

    Each element in the channel axis has its own standard deviation (\sigma)
    for Gaussian.
    Gaussian filter is adjusting its \sigma during training.

    Performs a 1D convolution on its input

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 3D tensor, with shape
        ``(batch_size, num_input_channels, input_length)``.

    filter_size : int or iterable of int
        An integer or a 1-element tuple specifying the size of the filters.
        This is the width of the filters that accomodate the Gaussian filters

    init_std : float
        The initial \sigma for the Gaussian filters

    stride : int or iterable of int
        An integer or a 1-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        An integer or a 1-element tuple results in symmetric zero-padding of
        the given size on both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

    W_logstd : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 1D tensor with shape
        ``(num_input_channels, )``.

        Note:
            The std is provided in log-scale, log(std).


    convolution : callable
        The convolution implementation to use. The
        `lasagne.theano_extensions.conv` module provides some alternative
        implementations for 1D convolutions, because the Theano API only
        features a 2D convolution implementation. Usually it should be fine
        to leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    """
    def __init__(self, incoming, filter_size,
                 init_std=5., W_logstd=None,
                 stride=1, pad=0,
                 nonlinearity=None,
                 convolution=conv1d_mc0, **kwargs):
        super(GaussianScan1DLayer, self).__init__(incoming, **kwargs)
        # convolution = conv1d_gpucorrmm_mc0
        # convolution = conv.conv1d_mc0
        # convolution = T.nnet.conv2d
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.filter_size = as_tuple(filter_size, 1)
        self.stride = as_tuple(stride, 1)
        self.convolution = convolution

        # if self.filter_size[0] % 2 == 0:
        #     raise NotImplementedError(
        #         'GaussianConv1dLayer requires odd filter size.')

        if pad == 'valid':
            self.pad = (0,)
        elif pad in ('full', 'same', 'strictsame'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 1, int)

        if W_logstd is None:
            init_std = np.asarray(init_std, dtype=floatX)
            W_logstd = init.Constant(np.log(init_std))
        # print(W_std)
        # W_std = init.Constant(init_std),
        self.num_input_channels = self.input_shape[1]
        # self.num_filters = self.num_input_channels
        self.W_logstd = self.add_param(W_logstd,
                                       (self.num_input_channels,),
                                       name="W_logstd",
                                       regularizable=False)
        self.W = self.make_gaussian_filter()

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        return (self.num_input_channels, self.num_input_channels,
                self.filter_size[0])

    def get_output_shape_for(self, input_shape):
        if self.pad == 'strictsame':
            output_length = input_shape[2]
        else:
            pad = self.pad if isinstance(self.pad, tuple) else (self.pad,)
            output_length = conv_output_length(
                input_shape[2],
                self.filter_size[0], self.stride[0], pad[0])

        return (input_shape[0], self.num_input_channels, output_length)

    def make_gaussian_filter(self):
        W_shape = self.get_W_shape()
        k = self.filter_size[0]
        k_low = int(np.floor(-(k-1)/2))
        k_high = k_low+k

        W_std = T.exp(self.W_logstd)
        std_array = T.tile(
            W_std.dimshuffle('x', 0, 'x'),
            (self.num_input_channels, 1, k)
        )

        x = np.arange(k_low, k_high).reshape((1, 1, -1))
        x = T.tile(
            x, (self.num_input_channels, self.num_input_channels, 1)
        ).astype(floatX)

        p1 = (1./(np.sqrt(2.*np.pi))).astype(floatX)
        p2 = np.asarray(2., dtype=floatX)
        gf = (p1/std_array)*T.exp(-x**2/(p2*(std_array**2)))
        # gf = gf.astype(theano.config.floatX)

        mask = np.zeros(W_shape)
        rg = np.arange(self.num_input_channels)
        mask[rg, rg, :] = 1
        mask = mask.astype(floatX)

        gf = gf*mask

        return gf

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1,) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            crop = self.filter_size[0] // 2
            conved = conved[:, :, crop:-crop or None]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = (0, 0)
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = (self.filter_size[0] // 2,
                       (self.filter_size[0] - 1) // 2)
            elif self.pad == 'strictsame':
                self.stride = (1,)
                border_mode = 'valid'
                kk = self.filter_size[0]-1
                rr = kk // 2
                ll = kk-rr
                pad = (ll, rr)
            else:
                border_mode = 'valid'
                pad = (self.pad[0], self.pad[0])
            if pad != (0, 0):
                input = padding.pad(input, [pad], batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0] + pad[1])
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        activation = conved

        return self.nonlinearity(activation)


class FixedGaussianScan1DLayer(GaussianScan1DLayer):
    """ 1D Fixed Gaussian filter
    Gaussian filter is not changing during the training

    Performs a 1D convolution on its input
    """
    def __init__(self, incoming, filter_size, init_std=5.,
                 stride=1, pad=0,
                 nonlinearity=None,
                 convolution=conv1d_mc0, **kwargs):
        super(GaussianScan1DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.filter_size = as_tuple(filter_size, 1)
        self.stride = as_tuple(stride, 1)
        self.convolution = convolution

        if pad == 'valid':
            self.pad = (0,)
        elif pad in ('full', 'same', 'strictsame'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 1, int)

        init_std = np.asarray(init_std, dtype=floatX)
        W_logstd = init.Constant(np.log(init_std))
        # print(W_std)
        # W_std = init.Constant(init_std),
        self.num_input_channels = self.input_shape[1]
        # self.num_filters = self.num_input_channels
        self.W_logstd = self.add_param(W_logstd,
                                       (self.num_input_channels,),
                                       name="W_logstd",
                                       regularizable=False,
                                       trainable=False)
        self.W = self.make_gaussian_filter()
