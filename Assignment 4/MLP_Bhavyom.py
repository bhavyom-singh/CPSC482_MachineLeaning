import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class MLP(nn.Module): # All models should inherit from nn.Module
    # This part can be changed based on the design decision.
    def __init__(self, num_input, hidden1_size, hidden2_size, hidden3_size, hidden4_size, num_classes): # Define our ANN structures here
        super(MLP, self).__init__()
        # nn.Linear(in_features, out_features, bias): y = w^Tx + bias
        self.hidden1 = nn.Linear(num_input, hidden1_size)    # connection between input and hidden layer1
        self.hidden2 = nn.Linear(hidden1_size, hidden2_size)
        self.hidden3 = nn.Linear(hidden2_size, hidden3_size)
        self.hidden4 = nn.Linear(hidden3_size, hidden4_size)
        self.output = nn.Linear(hidden4_size, num_classes)
        
        # The model structure can be also defined using "sequential" function
        # self.seq_linear=nn.Sequential(nn.Linear(num_input, hidden1_size),nn.RELU(),nn.Linear(hidden1_size, num_classes))
    # Define "forward" function to perform the computation for input x and return output(s).
    # The function name "forward" is required by Pytorch.
    def forward(self, x):
        # In this implementation, the activation function is reLU, but you can try other functions
        # torch.nn.functional modeule consists of all the activation functions and output functions
        h1_out = AF.relu(self.hidden1(x))
        h2_out = AF.relu(self.hidden2(h1_out))
        h3_out = AF.relu(self.hidden3(h2_out))
        h4_out = AF.relu(self.hidden4(h3_out))
        output = self.output(h4_out)
        # AF.softmax() is NOT needed when CrossEntropyLoss() is used as it already combines both LogSoftMax() and NLLLoss()
        
        # return self.seq_linear(x) # If the model structrue is define by sequential function.
        return output

def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    #print("Matrix for one image:")
    #print(images[1][0])
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()

# Training function
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


############################### ANN modeling #################################
### Convert the image into numbers: transforms.ToTensor()
# It separate the image into three color channels RGB and converts the pixels of each images to the brightness
# of the color in the range [0,255] that are scaled down to a range [0,1]. The image is now a Torch Tensor (array object)
### Normalize the tensor: transforms.Normalize() normalizes the tensor with mean (0.5) and stdev (0.5)
#+ You can change the mean and stdev values
print("------------------ANN modeling---------------------------")
transforms = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize((0.5,),(0.5,)),])
# PyTorch tensors are like NumPy arrays that can run on GPU
# e.g., x = torch.randn(64,100).type(dtype) # need to cast tensor to a CUDA datatype (dtype)
from torch.autograd import Variable
x = Variable
### Download and load the dataset from the torch vision library to the directory specified by root=''
# MNIST is a collection of 7000 handwritten digits (in images) split into 60000 training images and 1000 for testing 
# PyTorch library provides a clean data set. The following command will download training data in directory './data'
train_dataset=datasets.MNIST(root='dataset', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='dataset', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)
# You can use random_split function to splite a dataset
#from torch.utils.data.dataset import random_split
#train_data, val_data, test_data = random_split(train_dataset, [60,20,20])
### DataLoader will shuffle the training dataset and load the training and test dataset
mini_batch_size = 100 #+ You can change this mini_batch_size
# If mini_batch_size==100, # of training batches=6000/100=600 batches, each batch contains 100 samples (images, labels)
# DataLoader will load the data set, shuffle it, and partition it into a set of samples specified by mini_batch_size.
train_dataloader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches loaded for training: ", num_train_batches)
print("> Number of batches loaded for testing: ", num_test_batches)
### Let's display some images from the first batch to see what actual digit images look like
iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = False
if show_digit_image:
    show_some_digit_images(images)
### Create an object for the ANN model defined in the MLP class
# Architectural parameters: You can change these parameters except for num_input and num_classes
num_input = 28*28   # 28X28=784 pixels of image
num_classes = 10    # output layer
num_hidden1 = 784     # number of neurons at the first hidden layer
num_hidden2 = 200
num_hidden3 = 150
num_hidden4 = 80
# Randomly selected neurons by dropout_pr probability will be dropped (zeroed out) for regularization.
dropout_pr = 0.05
# MLP model
MLP_model=MLP(num_input, num_hidden1,num_hidden2, num_hidden3, num_hidden4, num_classes)
# Some model properties: 
# .state_dict(): a dictionary of trainable parameters with their current valeus
# .parameter(): a list of all trainable parameters in the model
# .train() or .eval(): setting training, testing mode
print("> MLP model parameters")
print(MLP_model.parameters)
# state_dict() maps each layer to its parameter tensor.
print ("> MLP model's state dictionary")
for param_tensor in MLP_model.state_dict():
    print(param_tensor, MLP_model.state_dict()[param_tensor].size())
#exit()

loss_func = nn.CrossEntropyLoss()
### Choose a gradient method
# model hyperparameters and gradient methods
# optim.SGD performs gradient descent and update the weigths through backpropagation.
num_epochs = 5
alpha = 0.02       # learning rate
gamma = 0.5        # momentum
# Stochastic Gradient Descent (SGD) is used in this program.
#+ You can choose other gradient methods (Adagrad, adadelta, Adam, etc.) and parameters
MLP_optimizer = optim.SGD(MLP_model.parameters(), lr=alpha, momentum=gamma) 
print("> MLP optimizer's state dictionary")
for var_name in MLP_optimizer.state_dict():
    print(var_name, MLP_optimizer.state_dict()[var_name])

### Train your networks
print("............Training MLP................")
# To turn on/off CUDA if I don't want to use it.
CUDA_enabled = True
is_MLP = True
train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, 
is_MLP, MLP_model, loss_func, MLP_optimizer)
print("............Testing MLP model................")
print("> Input digits:")
print(labels)
predicted_digits=test_ANN_model(device, CUDA_enabled, is_MLP, MLP_model, 
test_dataloader)
# print("> Predicted digits by MLP model")
# print(predicted_digits)


#### To save and load models and model's parameters ####
# To save and load model parameters
#"""
print("...Saving and loading model states and model parameters...")
torch.save(MLP_model.state_dict(), 'MLP_model_state_dict.pt')
loaded_MLP_model=MLP(num_input, num_hidden1,num_hidden2, num_hidden3, num_hidden4, num_classes)
loaded_MLP_model=MLP_model.load_state_dict(torch.load('MLP_model_state_dict.pt'))
torch.save(MLP_optimizer.state_dict(), 'MLP_optimizer_state_dict.pt')
loaded_MLP_optimizer = MLP_optimizer.load_state_dict(torch.load('MLP_optimizer_state_dict.pt'))
# To save and load a model
print("...Saving model...")
torch.save(MLP_model, 'MLP_model_NNIST.pt')
pretrained_model = torch.load('MLP_model_NNIST.pt')
#"""

