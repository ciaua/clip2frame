import theano
from lasagne.utils import floatX
import numpy as np
import pytest


class TestGaussianLayers:
    def gaussian_test_sets():
        for filter_size in [32, 256]:
            for init_std in [4, 16, 64]:
                yield (filter_size, init_std)

    def input_layer(self, output_shape):
        from lasagne.layers.input import InputLayer
        # from lasagne.layers import ReshapeLayer
        # input_layer_4d = InputLayer(output_shape)
        # input_layer_3d = ReshapeLayer(input_layer_4d, ([0], [1], -1))
        # return input_layer_3d
        return InputLayer(output_shape)

    def adaptivegaussian_layer(self, input_layer, filter_size, init_std):
        from jj.layers import GaussianScan1DLayer
        return GaussianScan1DLayer(
            input_layer,
            filter_size=filter_size,
            init_std=init_std,
            pad='strictsame'
        )

    def fixedgaussian_layer(self, input_layer, filter_size, init_std):
        from jj.layers import FixedGaussianScan1DLayer
        return FixedGaussianScan1DLayer(
            input_layer,
            filter_size=filter_size,
            init_std=init_std,
            pad='strictsame'
        )

    def make_numpy_gaussian_filter_v1(self, filter_size, std, axis=2):
        """
        test the case with one channel
        """

        k = filter_size
        k_low = int(np.floor(-(k-1)/2))
        k_high = k_low+k

        x = np.arange(k_low, k_high).reshape((1, 1, -1))
        # print(x)

        p1 = (1./(np.sqrt(2.*np.pi)))
        gf = (p1/std)*np.exp(-x**2/float(2*(std**2)))

        return gf

    def make_numpy_gaussian_filter_v2(self, filter_size, std, axis=2):
        """
        test the case with one channel
        """
        from scipy import stats

        dist = stats.norm(0, std)

        k = filter_size
        k_low = int(np.floor(-(k-1)/2))
        k_high = k_low+k

        x = np.arange(k_low, k_high).reshape((1, 1, -1))
        gf = dist.pdf(x)

        return gf

    def convolve_numpy_array(self, input, gaussian_filter):
        from scipy.signal import convolve
        return convolve(input, gaussian_filter, mode='same')

    @pytest.mark.parametrize(
        "filter_size, init_std", list(gaussian_test_sets()))
    def test_adaptivegaussian_layer(self, filter_size, init_std):
        input = floatX(np.ones((10, 1, 1000)))

        # test the case with one channel
        assert(input.shape[1] == 1)

        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)

        layer = self.adaptivegaussian_layer(input_layer, filter_size, init_std)
        layer_result = layer.get_output_for(input_theano).eval()

        # theano gaussian filter
        theano_gf = layer.W.eval()

        # numpy gaussian filter
        np_gf = self.make_numpy_gaussian_filter_v2(filter_size, init_std)

        numpy_result = self.convolve_numpy_array(input, np_gf)

        assert np.all(numpy_result.shape == layer.output_shape)
        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(theano_gf[0, 0, :], np_gf[0, 0, :])
        assert np.allclose(numpy_result, layer_result)

    @pytest.mark.parametrize(
        "filter_size, init_std", list(gaussian_test_sets()))
    def test_fixedgaussian_layer(self, filter_size, init_std):
        input = floatX(np.ones((10, 1, 1000)))

        # test the case with one channel
        assert(input.shape[1] == 1)

        input_layer = self.input_layer(input.shape)
        input_theano = theano.shared(input)

        layer = self.fixedgaussian_layer(input_layer, filter_size, init_std)
        layer_result = layer.get_output_for(input_theano).eval()

        # theano gaussian filter
        theano_gf = layer.W.eval()

        # numpy gaussian filter
        np_gf = self.make_numpy_gaussian_filter_v2(filter_size, init_std)

        numpy_result = self.convolve_numpy_array(input, np_gf)

        assert np.all(numpy_result.shape == layer.output_shape)
        assert np.all(numpy_result.shape == layer_result.shape)
        assert np.allclose(theano_gf[0, 0, :], np_gf[0, 0, :])
        assert np.allclose(numpy_result, layer_result)
