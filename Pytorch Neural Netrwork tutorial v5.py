# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:28:40 2025

@author: TechnoLEDs
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

path='C:\\Leo\\python scripts\\'

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root=path+"data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root=path+"data",
    train=False,
    download=True,
    transform=ToTensor(),
)

'''
Iterating and Visualizing the Dataset
=====================================
We can index Datasets manually like a list: training_data[index]. We use matplotlib 
to visualize some samples in our training data.
'''
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="Grays")
plt.show()

'''
Preparing your data for training with DataLoaders
=================================================
The Dataset retrieves our dataset’s features and labels one sample at a time. While 
training a model, we typically want to pass samples in “minibatches”, reshuffle the 
data at every epoch to reduce model overfitting, and use Python’s multiprocessing to 
speed up data retrieval.

We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, 
and supports automatic batching, sampling, shuffling and multiprocess data loading. 
Here we define a batch size of 64, i.e. each element in the dataloader iterable will 
return a batch of 64 features and labels.
'''
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

'''
Iterate through the DataLoader
We have loaded that dataset into the DataLoader and can iterate through the dataset 
as needed. Each iteration below returns a batch of train_features and train_labels 
(containing batch_size=64 features and labels respectively). Because we specified 
shuffle=True, after we iterate over all batches the data is shuffled
'''
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0].item()
plt.imshow(img, cmap="Grays")
plt.title(labels_map[label])
plt.show()
print(f"Label: {labels_map[label]}")


'''
Creating Models
===============
To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
We define the layers of the network in the __init__ function and specify how data will 
pass through the network in the forward function. To accelerate operations in the neural 
network, we move it to the accelerator such as CUDA, MPS, MTIA, or XPU. If the current 
accelerator is available, we will use it. Otherwise, we use the CPU.
'''
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

'''
Define the Class
================
We define our neural network by subclassing nn.Module, and initialize the neural network 
layers in __init__. Every nn.Module subclass implements the operations on input data 
in the forward method.
'''
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

'''
Loss Function
=============
When presented with some training data, our untrained network is likely not to give 
the correct answer. Loss function measures the degree of dissimilarity of obtained 
result to the target value, and it is the loss function that we want to minimize 
during training. To calculate the loss we make a prediction using the inputs of our 
given data sample and compare it against the true data label value.

Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks, 
and nn.NLLLoss (Negative Log Likelihood) for classification. nn.CrossEntropyLoss 
combines nn.LogSoftmax and nn.NLLLoss.

We pass our model’s output logits to nn.CrossEntropyLoss, which will normalize 
the logits and compute the prediction error.
'''
loss_fn = nn.CrossEntropyLoss()

'''
Optimizing the Model Parameters
===============================
To train a model, we need a loss function and an optimizer.
Optimization is the process of adjusting model parameters to reduce model error 
in each training step. Optimization algorithms define how this process is performed 
(in this example we use Stochastic Gradient Descent). All optimization logic is 
encapsulated in the optimizer object. Here, we use the SGD optimizer; additionally, 
there are many different optimizers available in PyTorch such as ADAM and RMSProp, 
that work better for different kinds of models and data.

We initialize the optimizer by registering the model’s parameters that need to be 
trained, and passing in the learning rate hyperparameter.
'''
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

'''
In a single training loop, the model makes predictions on the training dataset 
(fed to it in batches), and backpropagates the prediction error to adjust the 
model’s parameters.
'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # loss_ = loss
        
        '''
        Inside the training loop, optimization happens in three steps:
        Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients 
        by default add up; to prevent double-counting, we explicitly zero them at each iteration.
        
        Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits 
        the gradients of the loss w.r.t. each parameter.
        
        Once we have our gradients, we call optimizer.step() to adjust the parameters by 
        the gradients collected in the backward pass.
        '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Backpropagation
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
'''
We also check the model’s performance against the test dataset to ensure it is learning.
'''              
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct

'''
The training process is conducted over several iterations (epochs). During each epoch, 
the model learns parameters to make better predictions. We print the model’s accuracy 
and loss at each epoch; we’d like to see the accuracy increase and the loss decrease 
with every epoch.
'''
# training step
start_epoch = 0
end_epoch = start_epoch + 5
#epochs = 3
n_loss = []
n_acc = []
loss_ = 0
# for t in range(epochs):
for t in range(start_epoch, end_epoch):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    t_loss, t_acc = test(test_dataloader, model, loss_fn)
    
    n_loss.append(t_loss)
    n_acc.append(t_acc)   
print("Done!")
  
# plotting Loss and Accuracy
plt.figure(figsize = (11,5))
plt.subplot(1,2,1)
plt.plot(n_loss, label = 'Teste')
plt.xlabel('Épocas')
plt.ylabel('Função Custo')
plt.title('loss')
# plt.xticks()
# plt.yticks()
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.plot(n_acc, label = 'Teste')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('accuracy')
# plt.xticks()
# plt.yticks()
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

'''
https://pytorch.org/tutorials/beginner/saving_loading_models.html
Saving Model for inference
==========================
A common way to save a model is to serialize the internal state dictionary 
(containing the model parameters).
'''
torch.save(model.state_dict(), path + "model.pth")
print("Saved PyTorch Model State to model.pth")

'''
Saving General Checkpoint for Resuming Training
===============================================
When saving a general checkpoint, to be used for either inference or resuming training, 
you must save more than just the model’s state_dict. It is important to also save the 
optimizer’s state_dict, as this contains buffers and parameters that are updated as the 
model trains. Other items that you may want to save are the epoch you left off on, 
the latest recorded training loss, external torch.nn.Embedding layers, etc. 
'''
epoch = end_epoch
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': n_loss,
            'acc': n_acc,
            }, path + "model_train.pth")


'''
Loading Model for inference
===========================
The process for loading a model includes re-creating the model structure and 
loading the state dictionary into it.

In the code below, we set weights_only=True to limit the functions executed during 
unpickling to only those necessary for loading weights. Using weights_only=True is 
considered a best practice when loading weights.
'''
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(path + "model.pth", weights_only=True))
print("Loaded PyTorch Model State from model.pth")

'''
Remember that you must call model.eval() to set dropout and batch normalization 
layers to evaluation mode before running inference. Failing to do this will yield 
inconsistent inference results. If you wish to resuming training, call model.train() 
to ensure these layers are in training mode.
'''
test(test_dataloader, model, loss_fn)


'''
Loading General Checkpoint for Resuming Training
================================================
To load the items, first initialize the model and optimizer, then load the dictionary 
locally using torch.load(). From here, you can easily access the saved items by simply 
querying the dictionary as you would expect.
'''
checkpoint = torch.load(path + "model_train.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
n_loss = checkpoint['loss']
n_acc = checkpoint['acc']


# This model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# evaluate model:
r = random.randint(0, len(test_data)-1)
model.eval()
x, y = test_data[r][0], test_data[r][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    
    
