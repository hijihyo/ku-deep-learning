import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########


# TODO:
class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):

        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height)
        
        Returns
        -------
        out : (batch_size, num_filters, output_width, output_height)
        
        """

        out = self.convolute(x, self.W) + self.b
        return out

    def backprop(self, x, dLdy):
        """
        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height)
        dLdy : (num_batches, num_filters, output_width, output_height)
        
        Returns
        -------
        dLdx.shape : (num_batches, num_input_channels, input_width, input_height)
        dLdW.shape : (num_filters, num_input_channels, filter_width, filter_height)
        dLdb.shape : (1, num_filter, 1, 1)
        
        """

        # dLdx
        dLdy_padded = np.pad(
            dLdy, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
        temp = np.swapaxes(self.W, 0, 1)
        temp = np.flip(temp, axis=temp.ndim-2)
        filters = np.flip(temp, axis=temp.ndim-1)
        dLdx = self.convolute(dLdy_padded, filters)

        # dLdW
        dLdy_padded = np.pad(
            dLdy, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='constant')
        dLdy_padded = np.swapaxes(dLdy_padded, 0, 1)
        filters = np.swapaxes(x, 0, 1)
        dLdW = self.convolute(dLdy_padded, filters)
        dLdW = np.flip(dLdW, axis=temp.ndim-2)
        dLdW = np.flip(dLdW, axis=temp.ndim-1)

        # dLdb
        temp = dLdy.sum(axis=0).reshape((1,) + dLdy.shape[1:])
        temp = temp.sum(axis=2).sum(axis=2)
        dLdb = temp.reshape(temp.shape[:2] + (1, 1))

        return dLdx, dLdW, dLdb

    def convolute(self, x, filters):
        """
            Parameters
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            - filters.shape: (num_filters, num_input_channels, filter_width, filter_height)
            Returns
            - out.shape: (num_batches, num_filters, output_width, output_height)
        """

        num_batches, _, _, _ = x.shape
        num_filters, _, _, _ = filters.shape

        x_prime = view_as_windows(x, (1,) + filters.shape[1:])
        x_prime = x_prime.reshape(x_prime.shape[:5] + (-1,))

        result = np.zeros((num_batches, num_filters) + x_prime.shape[2:4])
        for i in range(num_filters):
            temp = x_prime.dot(filters[i].reshape(-1))
            temp = temp.squeeze(axis=1)
            result[:, i, :, :] = temp.squeeze(axis=temp.ndim-1)

        return result

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

# TODO:
class nn_max_pooling_layer:

    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height)
        
        Returns
        -------
        out : (batch_size, input_channel_size, output_width, output_height)
        
        """

        out = block_reduce(x, (1, 1, 2, 2), np.max)
        return out

    # TODO: 코드를 더 개선시킬 수 있을 것 같음
    def backprop(self, x, dLdy):
        """
        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height)
        dLdy.shape : (batch_size, input_channel_size, output_width, output_height)
        
        Returns
        -------
        dLdx : (batch_size, input_channel_size, input_width, input_height)
        
        """

        pool = self.forward(x)
        pool = pool.repeat(2, axis=3)
        pool = pool.repeat(2, axis=2)
        dydx = np.equal(x, pool)
        dLdy = dLdy.repeat(2, axis=3).repeat(2, axis=2)
        dLdx = dLdy * dydx

        return dLdx


##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:
    """ A fully-connected layer in neural networks

    Attributes
    ----------
    W: (output_size, input_size)
    b: (output_size)

    """

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(
            0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b = 0.01+np.zeros((output_size))

    def forward(self, x):
        """ Calculates outputs of the layer in forward

        Parameters
        ----------
        x: (batch_size, input_channel_size, input_width, input_height)

        Returns
        -------
        result: (batch_size, output_size)

        """

        # (batch_size, input_channel_size, input_width, input_height) -> (batch_size, input_size)
        x_flat = x.reshape((x.shape[0], -1))
        result = x_flat @ (self.W.T) + self.b.T

        # (batch_size, output_size)
        return result

    def backprop(self, x, dLdy):
        """ Calculates (downstream) gradients of the layer

        Parameters
        ----------
        x: (batch_size, num_filters, input_width, input_height)
        dLdy: (batch_size, output_size)

        Returns
        -------
        dLdx : (batch_size, num_filters, input_width, input_height)
        dLdW : (output_size, input_size)
        dLdb : (output_size)

        References
        ----------
        Backpropagation for a Linear Layer
        - https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html

        """

        # (batch_size, input_channel_size, input_width, input_height) -> (batch_size, input_size)
        x_flat = x.reshape((x.shape[0], -1))

        # (batch_size, output_size) X (output_size, input_size)
        dLdx_flat = dLdy @ self.W
        # (batch_size, input_size) -> (batch_size, num_filters, input_width, input_height)
        dLdx = dLdx_flat.reshape(x.shape)

        # (output_size, batch_size) X (batch_size, input_size)
        dLdW = dLdy.T @ x_flat

        # (output_size)
        dLdb = dLdy.sum(axis=0)

        return dLdx, dLdW, dLdb

    def update_weights(self, dLdW, dLdb):
        # parameter update
        self.W = self.W + dLdW
        self.b = self.b + dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    """ An ReLU-activation layer in neural networks

    """

    # performs ReLU activation
    def __init__(self):
        pass

    def forward(self, x):
        """ Calculates outputs of the layer in forward

        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height) || (batch_size, input_size)
        
        Returns
        -------
        out : same as x
        
        """

        result = np.where(x >= 0, x, 0)
        return result

    def backprop(self, x, dLdy):
        """ Calculates (downstream) gradients of the layer

        Parameters
        ----------
        x : (batch_size, input_channel_size, input_width, input_height) || (batch_size, input_size)
        dLdy : same as x
        
        Returns
        -------
        dLdx : same as x
        
        """

        dLdx = np.where(x > 0, dLdy, 0)
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:
    """ An softmax layer in neural networks

    """

    def __init__(self):
        pass

    def softmax(self, x):
        """ Calculates softmax of the given value, vector, or matrix

        Parameters
        ----------
        x: (batch_size, input_size)

        Returns
        -------
        result: (batch_size, input_size)

        References
        ----------
        Numercially stable softmax
        - https://stackoverflow.com/questions/42599498/numercially-stable-softmax

        """

        # numerically stable softmax
        z = x - x.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, x):
        """ Calculates outputs of the layer in forward

        Parameters
        ----------
        x: (batch_size, input_size)

        Returns
        -------
        result: (batch_size, output_size) where output_size = input_size

        """

        softmax = self.softmax(x)

        # (batch_size, output_size) where output_size = input_size
        return softmax

    def backprop(self, x, dLdy):
        """ Calculates (downstream) gradients of the layer

        Parameters
        ----------
        x: (batch_size, input_size)
        dLdy: (batch_size, output_size) where output_size = input_size

        Returns
        -------
        dLdx: (batch_size, input_size) where input_size = output_size

        References
        ----------
        Vectorized softmax gradient
        - https://stackoverflow.com/questions/59286911/vectorized-softmax-gradient

        """

        y = self.forward(x)

        dydx = \
            np.expand_dims(y, axis=2) * np.eye(x.shape[1]) \
            - np.expand_dims(y, axis=2) @ np.expand_dims(y, axis=1)

        # (batch_size, output_size, output_size) X (batch_size, output_size, 1)
        dLdx = dydx @ np.expand_dims(dLdy, axis=2)

        # (batch_size, input_size) where input_size = output_size
        return dLdx.squeeze()

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:
    """ A cross entropy layer in neural networks

    """

    def __init__(self):
        pass

    def forward(self, x, y):
        """ Calculates outputs of the layer in forward

        Parameters
        ----------
        x: (batch_size, input_size)
        y: (batch_size)

        Returns
        -------
        result: number

        """

        batch_size, input_size = x.shape

        index = y.reshape(-1, 1) == np.arange(0, input_size).reshape(1, -1) # (batch_size, input_size)
        log_loss = np.log(x) # (batch_size, input_size)
        avg_loss = - np.sum(log_loss[index]) / batch_size

        return avg_loss

    def backprop(self, x, y):
        """ Calculates (downstream) gradients of the layer

        Parameters
        ----------
        x: (batch_size, input_size)
        y: (batch_size, 1)

        Returns
        -------
        dLdx: (batch_size, input_size)

        """

        batch_size, input_size = x.shape

        index = y.reshape(-1, 1) != np.arange(0, input_size).reshape(1, -1) # (batch_size, input_size)
        dLdx = - 1 / (batch_size * x) # (batch_size, input_size)
        dLdx[index] = 0

        return dLdx
