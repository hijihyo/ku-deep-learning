import sys
import os
import matplotlib.pyplot as plt
import numpy as np

class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))
    
    ######
    ## Q1
    def forward(self,x):
        # self.W: (O, I) / self.b: (O, 1)
        # x: (B, I) / s: (B, O)
        s = x @ (self.W.T) + self.b.T
        return s
    
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        # x: (B, I) / dLdy: (B, O)
        # dLdW: (O, I) / dLdb: (O, 1) / dLdx: (B, I)
        (batch_size, input_size) = x.shape
        (_, output_size) = dLdy.shape

        dydW = x # (B, I)
        dLdW = (dLdy.T) @ dydW

        dydb = np.zeros((batch_size, output_size, output_size))
        dydb[np.arange(batch_size),:,:] = np.eye(output_size)
        dLdb = np.matmul(dydb, dLdy.reshape((batch_size, output_size, 1)))
        dLdb = dLdb.reshape((batch_size, output_size)).sum(axis=0)

        dydx = self.W # (O, I)
        dLdx = dLdy @ dydx

        return dLdW,dLdb.reshape(1, output_size),dLdx

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        # x: (B, I)
        y = 1 / (1 + np.exp(-x))
        return y
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        # x: (B, I) / dLdy: (B, I)
        (batch_size, input_size) = x.shape

        # temp: (B, I)
        exp_neg_x = np.exp(-x)
        temp = exp_neg_x / ((1 + exp_neg_x) ** 2)
        
        # dydx: (B, I, I)
        dydx = np.zeros((batch_size, input_size, input_size))
        for i in range(batch_size):
            for j in range(input_size):
                dydx[i, j, j] = temp[i][j]
        
        dLdy = dLdy.reshape(batch_size, input_size, 1)
        dLdx = np.matmul(dydx, dLdy).reshape(batch_size, input_size)
        return dLdx


class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        # x: (B, I)
        (batch_size, input_size) = x.shape

        exp_x = np.exp(x)
        softmax_x = exp_x / exp_x.sum(axis=1).reshape(batch_size, 1)
        return softmax_x
    
    ######
    ## Q6
    def backprop(self,x,dLdy):
        # x: (B, I) / dLdy: (B, I)
        (batch_size, input_size) = x.shape

        # softmax_x: (B, I)
        softmax_x = self.forward(x)
        
        # dydx: (B, I, I)
        dydx = np.zeros((batch_size, input_size, input_size))
        for i in range(batch_size):
            for j in range(input_size):
                for k in range(input_size):
                    if j == k: dydx[i, j, k] = softmax_x[i][j] * (1 - softmax_x[i][j])
                    else: dydx[i, j, k] = - softmax_x[i][j] * softmax_x[i][k]
        
        dLdy = dLdy.reshape(batch_size, input_size, 1)
        dLdx = np.matmul(dydx, dLdy).reshape(batch_size, input_size)
        return dLdx

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    def forward(self,x,y):
        # x: (B, I) / y: (B, 1)
        n = y.size
        loss_i = np.log(np.array([x[i, y[i]] for i in range(n)]))
        avg_loss = - np.sum(loss_i) / n
        return avg_loss
        
    ######
    ## Q8
    def backprop(self,x,y):
        # x: (B, I) / y: (B, 1)
        # dLdx: (B, I)
        (batch_size, _) = y.shape

        dLdx = - 1 / (batch_size * x)
        for i in range(batch_size):
            dLdx[i][1-y[i]] = 0
        return dLdx

# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d=5

# number of test runs
num_test=40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr = 1.0 # float(sys.argv[1])
num_gd_step = 5000 # int(sys.argv[2])

# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=False

# set this True if want to plot loss over gradient descent iteration
show_loss=True

################
# create training data
################

m_d1 = (0, 0)
m_d2 = (1, 1)
m_d3 = (0, 1)
m_d4 = (1, 0)

sig = 0.05
s_d1 = sig ** 2 * np.eye(2)

d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)
d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

# training data, and has shape (4*num_d,2)
x_train_d = np.vstack((d1, d2, d3, d4))
# training data lables, and has shape (4*num_d,1)
y_train_d = np.vstack((np.zeros((2 * num_d, 1), dtype='uint8'), np.ones((2 * num_d, 1), dtype='uint8')))

if (show_train_data):
    plt.grid()
    plt.scatter(x_train_d[range(2 * num_d), 0], x_train_d[range(2 * num_d), 1], color='b', marker='o')
    plt.scatter(x_train_d[range(2 * num_d, 4 * num_d), 0], x_train_d[range(2 * num_d, 4 * num_d), 1], color='r',
                marker='x')
    plt.show()

################
# create layers
################

# hidden layer
# linear layer
layer1 = nn_linear_layer(input_size=2, output_size=4, )
# activation layer
act = nn_activation_layer()

# output layer
# linear
layer2 = nn_linear_layer(input_size=4, output_size=2, )
# softmax
smax = nn_softmax_layer()
# cross entropy
cent = nn_cross_entropy_layer()

# variable for plotting loss
loss_out = np.zeros((num_gd_step))

################
# do training
################

for i in range(num_gd_step):
    
    # fetch data
    x_train = x_train_d
    y_train = y_train_d
    
    # create one-hot vectors from the ground truth labels
    y_onehot = np.zeros((batch_size, num_class))
    y_onehot[range(batch_size), y_train.reshape(batch_size, )] = 1
    
    ################
    # forward pass
    
    # hidden layer
    # linear
    l1_out = layer1.forward(x_train)
    # activation
    a1_out = act.forward(l1_out)
    
    # output layer
    # linear
    l2_out = layer2.forward(a1_out)
    # softmax
    smax_out = smax.forward(l2_out)
    # cross entropy loss
    loss_out[i] = cent.forward(smax_out, y_train)
    
    ################
    # perform backprop
    # output layer
    # cross entropy
    b_cent_out = cent.backprop(smax_out, y_train)
    # softmax
    b_nce_smax_out = smax.backprop(l2_out, b_cent_out)
    
    # linear
    b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)
    
    # backprop, hidden layer
    # activation
    b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
    # linear
    b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)
    
    ################
    # update weights: perform gradient descent
    layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2.T * lr)
    layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1.T * lr)
    
    if (i + 1) % 2000 == 0:
        print('gradient descent iteration:', i + 1)

# set show_loss to True to plot the loss over gradient descent iterations
if (show_loss):
    plt.figure(1)
    plt.grid()
    plt.plot(range(num_gd_step), loss_out)
    plt.xlabel('number of gradient descent steps')
    plt.ylabel('cross entropy loss')
    plt.show()

################
# training done
# now testing

num_test = 100

for j in range(num_test):
    
    predicted = np.ones((4,))
    
    # dispersion of test data
    sig_t = 1e-2
    
    # generate test data
    # generate 4 samples, each sample nearby (1,1), (0,0), (1,0), (0,1) respectively
    t11 = np.random.multivariate_normal((1,1), sig_t**2*np.eye(2), 1)
    t00 = np.random.multivariate_normal((0,0), sig_t**2*np.eye(2), 1)
    t10 = np.random.multivariate_normal((1,0), sig_t**2*np.eye(2), 1)
    t01 = np.random.multivariate_normal((0,1), sig_t**2*np.eye(2), 1)
    
    # predicting label for test sample nearby (1,1)
    l1_out = layer1.forward(t11)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)', smax_out, 'predicted label:', int(predicted[0]))
    
    # predicting label for test sample nearby (0,0)
    l1_out = layer1.forward(t00)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)', smax_out, 'predicted label:', int(predicted[1]))
    
    # predicting label for test sample nearby (1,0)
    l1_out = layer1.forward(t10)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)', smax_out, 'predicted label:', int(predicted[2]))
    
    # predicting label for test sample nearby (0,1)
    l1_out = layer1.forward(t01)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)', smax_out, 'predicted label:', int(predicted[3]))
    
    print('total predicted labels:', predicted.astype('uint8'))
    
    accuracy += (predicted[0] == 0) & (predicted[1] == 0) & (predicted[2] == 1) & (predicted[3] == 1)
    
    if (j + 1) % 10 == 0:
        print('test iteration:', j + 1)

print('accuracy:', accuracy / num_test * 100, '%')
# print(f"accuracy: {accuracy / num_test * 100} % for lr={ lr }, ns={ num_gd_step }")






