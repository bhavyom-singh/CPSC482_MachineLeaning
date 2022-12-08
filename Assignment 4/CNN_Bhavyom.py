import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import cv2

### CNN architecture
class CNN(nn.Module):
    # The probability of dropout, number of hidden nodes, number of output classes
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()
        # Conv2d is for two dimensional convolution (which means input image is 2D)
        # Conv2d(in_channels=, out_channels=, kernel_size=, stride=1, padding=0)
        # in_channels=1 if grayscale, 3 for color; out_channels is # of output channels (or # of kernels)
        # Ouput size for W from CONV = (W-F+2P)/S+1 where W=input_size, F=filter_size, S=stride, P=padding
        # max_pool2d(input_tensor, kernel_size, stride=None, padding=0) for 2D max pooling
        # Output size for W from POOL = (W-F)/S+1 where S=stride=dimension of pool
        # K is # of channels for convolution layer; D is # of channels for pooling layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # K=D=10, output_size W=(28-5)/1+1=24 (24x24), (default S=1)
        self.pool1 = nn.MaxPool2d(4, 4) # W = (24-4)/1+1=21 (21x21), S=4 (pool dimension) since no overlapping regions
        self.dropout_conv1 = nn.Dropout2d(dropout_pr) # to avoid overfitting by dropping some nodes

        self.conv2 = nn.Conv2d(21, 17, kernel_size=5) # K=D=10, output_size W=(21-5)/1+1=17 (17x17), (default S=1)
        self.pool2 = nn.MaxPool2d(4, 4) # W = (17-4)/1+1=14 (12x12), S=1 (pool dimension) since no overlapping regions
        self.dropout_conv2 = nn.Dropout2d(dropout_pr) # to avoid overfitting by dropping some nodes

        self.conv3 = nn.Conv2d(14, 10, kernel_size=5) # K=D=10, output_size W=(14-5)/1+1=10 (10x10), (default S=1)
        self.pool3 = nn.MaxPool2d(4, 4) # W = (10-4)/1+1=7 (7x7), S=1 (pool dimension) since no overlapping regions
        self.dropout_conv3 = nn.Dropout2d(dropout_pr) # to avoid overfitting by dropping some nodes
        #+ You can add more convolutional and pooling layers
        # Fully connected layer after convolutional and pooling layers
        # self.num_flatten_nodes = 10*6*6 # Flatten nodes from 10 channels and 6*6 pool_size = 10*6*6=360
        self.num_flatten_nodes = 360 # Flatten nodes from 10 channels and 6*6 pool_size = 10*6*6=360
        self.hidden1 = 200
        self.hidden2 = 150
        self.hidden3 = 100
        self.fc1 = nn.Linear(self.num_flatten_nodes, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        #+ You can add more hidden layers here if necessary
        self.out = nn.Linear(self.hidden3, num_classes) # the output nodes are 10 classes (10 digits)
        
    def forward(self, x):
        out = AF.relu(self.pool1(self.conv1(x)))
        out = AF.relu(self.dropout_conv1(out))
        # out = AF.relu(self.pool2(self.conv2(x)))
        # out = AF.relu(self.dropout_conv2(out))
        # out = AF.relu(self.pool3(self.conv3(x)))
        # out = AF.relu(self.dropout_conv3(out))
        out = out.view(-1, self.num_flatten_nodes) # flattening
        out = AF.relu(self.fc1(out))        
        out = AF.relu(self.fc2(out))        
        out = AF.relu(self.fc3(out))
        # Apply dropout for the randomly selected nodes by zeroing out before output during training
        out = AF.dropout(out)
        output = self.out(out)
        return output

# To display some images
def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()


def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, is_MLP, ANN_model, loss_func, optimizer):
    if (device.type == 'cuda' and CUDA_enabled):
        print("...Modeling using GPU...")        
        ANN_model = ANN_model.to(device=device) # sending to whaever device (for GPU acceleration)
        print("Memory allocated:", torch.cuda.memory_allocated(cuda_id))
        print("Memory reserved:", torch.cuda.memory_reserved(cuda_id))
        if (next(ANN_model.parameters()).is_cuda):
            print("ANN_model is on GPU...")
    else:
        print("...Modeling using CPU...")
    train_losses = []
        
    for epoch_cnt in range(num_epochs):
        ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(training_data):
            # Each batch contain batch_size (100) images, each of which 1 channel 28x28
            # print(images.shape) # the shape of images=[100,1,28,28]
            # So, we need to flatten the images into 28*28=784
            # -1 tells NumPy to flatten to 1D (784 pixels as input) for batch_size images
            if (is_MLP):
                # the size -1 is inferred from other dimensions
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            optimizer.zero_grad() # set the cumulated gradient to zero
            output = ANN_model(images) # feedforward images as input to the network
            loss = loss_func(output, labels) # computing loss
            #print("Loss: ", loss)
            #print("Loss item: ", loss.item())
            train_losses.append(loss.item())
            # PyTorch's Autograd engine (automatic differential (chain rule) package) 
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation
            # Display the training status
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses
# Testing function
def test_ANN_model(device, CUDA_enabled, is_MLP, ANN_model, testing_data):
    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients    
    predicted_digits=[]
    # torch.no_grad() deactivates Autogra engine (for weight updates). This help run faster
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)
            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)
            
            output = ANN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
        print("> Number of samples=", num_samples, "number of correct prediction=",num_correct, "accuracy=", accuracy)
    return predicted_digits

########################### Checking GPU and setup #########################
### CUDA is a parallel computing platform and toolkit developed by NVIDIA. 
# CUDA enables parallelize the computing intensive operations using GPUs.
# In order to use CUDA, your computer needs to have a CUDA supported GPU and install the CUDA Toolkit
# Steps to verify and setup Pytorch and CUDA Toolkit to utilize your GPU in your machine:
# (1) Check if your computer has a compatible GPU at https://developer.nvidia.com/cuda-gpus
# (2) If you have a GPU, continue to the next step, else you can only use CPU and ignore the rest steps.
# (3) Downloaded the compatible Pytorch version and CUDA version, refer to https://pytorch.org/get-started/locally/
# Note: If Pytorch and CUDA versions are not compatible, Pytorch will not be able to recognize your GPU
# (4) The following codes will verify if Pytorch is able to recognize the CUDA Toolkit:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    # #"""
    # # Maxtrix multiplication C = A*B
    # d = 20000 # Try to change this dimension and see the difference
    # print("CPU and GPU comparison for ", d, "x", d, "matrix multiplication C = A*B")
    # # using numpy
    # start_time = time.time()
    # A = np.random.rand(d, d).astype(np.float32)
    # B = np.random.rand(d, d).astype(np.float32)
    # C = A.dot(B)
    # print("NumPy took: ", (time.time() - start_time), "seconds")
    # # using Torch with GPU
    # start_time = time.time()
    # A = torch.rand(d, d).cuda()
    # B = torch.rand(d, d).cuda()
    # C = torch.mm(A, B)
    # print("Torch.CUDA took: ", (time.time() - start_time), "seconds\n")
    # #"""
    print("GPU will be utilized for computation.")
else:
    print("CUDA is NOT supported in your machine. Only CPU will be used for computation.")
#exit()

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
from torch.autograd import Variable
x = Variable

train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)

mini_batch_size = 100

train_dataloader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches loaded for training: ", num_train_batches)
print("> Number of batches loaded for testing: ", num_test_batches)


iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = False
if show_digit_image:
    show_some_digit_images(images)

num_input = 28*28   # 28X28=784 pixels of image
num_classes = 10    # output layer
num_hidden = 10     # number of neurons at the first hidden layer
# Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.
dropout_pr = 0.05

# CNN model
CNN_model = CNN(dropout_pr, num_hidden, num_classes)
print("> CNN model parameters")
print(CNN_model.parameters)
### Define a loss function: You can choose other loss functions
loss_func = nn.CrossEntropyLoss()
### Choose a gradient method
# model hyperparameters and gradient methods
# optim.SGD performs gradient descent and update the weigths through backpropagation.
num_epochs = 5
alpha = 0.01       # learning rate
gamma = 0.5        # momentum

# CNN optimizer
CNN_optimizer = optim.SGD(CNN_model.parameters(), lr=alpha, momentum=gamma)
CUDA_enabled = True
print("............Training CNN................")
is_MLP = False
train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, is_MLP, CNN_model, loss_func, CNN_optimizer)
print("............Testing CNN model................")
predicted_digits=test_ANN_model(device, CUDA_enabled, is_MLP, CNN_model, test_dataloader)
# print("> Predicted digits by CNN model")
# print(predicted_digits)

# def LoadImage(filePath):
#         # Load sample image
#         file = filePath
#         test_image = cv2.imread(file)

#         # Preview sample image
#         plt.imshow(test_image, cmap='gray')

#         # Format Image
#         img_resized = cv2.resize(test_image, (25, 10), interpolation=cv2.INTER_LINEAR)
#         img_resized = cv2.bitwise_not(img_resized)

#         # Preview reformatted image
        

#         return img_resized
# def predict():
#     img = LoadImage(r'D:/AI Projects/FinalProjectCandidate1/one.png')
#     print(img.shape)
#     print(img.shape[2])
#     if (img.shape[2] == 3):        
#         img = img.reshape(img.shape[0], -1)
#     #flattened_img = img.reshape(28, 28)
#     transformFunc = transforms

#     tensorImg = transformFunc(img)
#     output=CNN_model(tensorImg.to(device)) #applying the model we have built
#     print(f'output : {output}')
#     #labels=labels
#     prediction=np.argmax(output)
#     print(prediction)

# print(f"prediction of the file: {predict()}")