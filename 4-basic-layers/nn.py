import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######


class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(
            0,
            std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
            (num_filters, in_ch_size, filter_width, filter_height)
        )
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        # If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        """
            Parameters:
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            Variables:
            - self.W.shape: (num_filters, num_input_channels, filter_width, filter_height)
            - self.b.shape: (1, num_filters, 1, 1)
            Returns:
            - out.shape: (num_batches, num_filters, output_width, output_height)
        """

        out = self.convolute(x, self.W) + self.b
        return out

    #######
    # Q2. Complete this method
    #######

    def backprop(self, x, dLdy):
        """
            Parameters:
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            - dLdy.shape: (num_batches, num_filters, output_width, output_height)
            Variables:
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            - dydb.shape: (num_batches, num_filters, 1, 1)
            Returns:
            - dLdx.shape: (num_batches, num_input_channels, input_width, input_height)
            - dLdW.shape: (num_filters, num_input_channels, filter_width, filter_height)
            - dLdb.shape: (1, num_filter, 1, 1)
        """

        # num_batches, num_input_channels, input_width, input_height = x.shape
        # num_filters, _, filter_width, filter_height = self.W.shape

        # dLdx
        dLdy_padded = np.pad(dLdy, ((0,0),(0,0),(2,2),(2,2)), mode='constant')
        temp = np.swapaxes(self.W, 0, 1)
        temp = np.flip(temp, axis=temp.ndim-2)
        filters = np.flip(temp, axis=temp.ndim-1)
        dLdx = self.convolute(dLdy_padded, filters)

        # dLdW
        dLdy_padded = np.pad(dLdy, ((0,0),(0,0),(2,2),(2,2)), mode='constant')
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

    #######
    # If necessary, you can define additional class methods here
    #######
    def convolute(self, X, filters):
        """
            Parameters
            - X.shape: (num_batches, num_input_channels, input_width, input_height)
            - filters.shape: (num_filters, num_input_channels, filter_width, filter_height)
            Returns
            - out.shape: (num_batches, num_filters, output_width, output_height)
        """

        num_batches, _, _, _ = X.shape
        num_filters, _, _, _ = filters.shape
        
        X_prime = view_as_windows(X, (1,) + filters.shape[1:])
        X_prime = X_prime.reshape(X_prime.shape[:5] + (-1,))
        
        result = np.zeros((num_batches, num_filters) + X_prime.shape[2:4])
        for i in range(num_filters):
            temp = X_prime.dot(filters[i].reshape(-1))
            temp = temp.squeeze(axis=1)
            result[:, i, :, :] = temp.squeeze(axis=temp.ndim-1)

        return result


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        # If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        """
            Parameters:
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            Returns:
            - out.shape: (num_batches, num_input_channels, output_width, output_height)
        """

        X_prime = view_as_windows(x, (1, 1, self.pool_size, self.pool_size), step=(1, 1, 2, 2))
        X_prime = X_prime.reshape(X_prime.shape[:x.ndim] + (-1,))
        out = np.max(X_prime, axis=x.ndim)

        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        """
            Parameters:
            - x.shape: (num_batches, num_input_channels, input_width, input_height)
            - dLdy.shape: (num_batches, num_input_channels, output_width, output_height)
            Variables:
            - dydx.shape: (num_batches, num_input_channels, output_width, output_height, num_batches, num_input_channels, input_width, input_height)
            Returns:
            - dLdx.shape: (num_batches, num_input_channels, output_width, output_height)
        """

        pool = self.forward(x)
        pool = pool.repeat(2, axis=3)
        pool = pool.repeat(2, axis=2)
        dydx = np.equal(x, pool)
        dLdy = dLdy.repeat(2, axis=3).repeat(2, axis=2)
        dLdx = dLdy * dydx

        return dLdx

    #######
    # If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(
        filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(
        0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(
        0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(
        0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) /
                         exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')
