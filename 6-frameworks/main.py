import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class BasicBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()

        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding)

        self.batch_norm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding)
    
    def forward(self, x):
        h_bn1 = self.batch_norm1(x)
        h_lu1 = nn.functional.relu(h_bn1)
        h_cv1 = self.conv1(h_lu1)

        h_bn2 = self.batch_norm2(h_cv1)
        h_lu2 = nn.functional.relu(h_bn2)
        h_cv2 = self.conv2(h_lu2)

        out = x + h_cv2
        return out

class ModifiedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride1, stride2, padding):
        super(ModifiedBlock, self).__init__()

        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride1, padding)

        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride2, padding)

        self.pool = nn.Conv2d(in_channels, out_channels, 1, 2)
    
    def forward(self, x):
        h_bn1 = self.batch_norm1(x)
        h_lu1 = nn.functional.relu(h_bn1)
        h_cv1 = self.conv1(h_lu1)

        h_bn2 = self.batch_norm2(h_cv1)
        h_lu2 = nn.functional.relu(h_bn2)
        h_cv2 = self.conv2(h_lu2)

        h_pool = self.pool(h_lu1)

        out = h_pool + h_cv2
        return out

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1)

        stage1_list = [ BasicBlock(64, 3, 1, 1) for i in range(nblk_stage1) ]
        self.stage1 = nn.Sequential(*stage1_list)

        stage2_list = [ ModifiedBlock(64, 128, 3, 2, 1, 1) ] + [ BasicBlock(128, 3, 1, 1) for i in range(nblk_stage2-1) ]
        self.stage2 = nn.Sequential(*stage2_list)

        stage3_list = [ ModifiedBlock(128, 256, 3, 2, 1, 1) ] + [ BasicBlock(256, 3, 1, 1) for i in range(nblk_stage3-1) ]
        self.stage3 = nn.Sequential(*stage3_list)

        stage4_list = [ ModifiedBlock(256, 512, 3, 2, 1, 1) ] + [ BasicBlock(512, 3, 1, 1) for i in range(nblk_stage4-1) ]
        self.stage4 = nn.Sequential(*stage4_list)

        self.fully = nn.Linear(512, 10)
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################

    ########################################
    # You can define whatever methods
    ########################################
    
    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################
        h_cv0 = self.conv(x)
        h_stage1 = self.stage1(h_cv0)
        h_stage2 = self.stage2(h_stage1)
        h_stage3 = self.stage3(h_stage2)
        h_stage4 = self.stage4(h_stage3)

        # print(h_stage4.shape)
        h_avg = nn.functional.avg_pool2d(h_stage4, kernel_size=4, stride=4).squeeze()
        # print(h_avg.shape)
        out = self.fully(h_avg)

        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 64

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        optimizer.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net.forward(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')
