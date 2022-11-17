#basic net from- https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#constants
BATCH_SIZE = 100




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
#prepar the data to train/ test by shuffle, load in batch instead of one by one etc.
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0),
}
loaders


# creat the convunetional neural network
class CNN(nn.Module):
    def __init__(self):
        #the base class get the "son" name and the module
        super(CNN, self).__init__()
        #define the first convolution
        # img size= (28*28)->2 padding ->(32*32)->conv kernal 5->(28*28)->
        # max pooling kernal 2->(14*14)
        #Cin=1, Cout=16
        self.conv1 = nn.Sequential(         
            #add convolution
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),              
            #wrap the convolution with relu                
            nn.ReLU(),                     
            #wrap the relu with max pooling 
            nn.MaxPool2d(kernel_size=2),    
        )
        #define the second convolution
        # img size= (14*14)->2 padding ->(18*18)->conv kernal 5->(14*14)->
        # max pooling kernal 2->(7*7)
        #Cin=16, Cout=32
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer
        #img size (7*7)-> flatten the img->(1568*1)
        #  Cin 32 , output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    #combine all the "blocks" 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7=1568)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
cnn = CNN()
#print( cnn )

#define the "cross entropy" loss function
loss_func = nn.CrossEntropyLoss()
#print(loss_func)

# take the betas from our CNN class
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
#print(optimizer)

num_epochs = 10
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        
        pass  
    
    pass
train(num_epochs, cnn, loaders)

def test():
    # Call the base class to go to test the model
    cnn.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    
pass
test()
