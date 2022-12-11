#TAKEN FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#d75c
#autoencoder using cnn

import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/MNIST/autoencoder_tensorboard')

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)
print("len(train_dataset)= ",m,"len(test_dataset)",len(test_dataset))
print(test_dataset.targets)
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)



class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

def plot_ae_outputs(encoder,decoder,n=10):
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show() 

### task one and two creat image from random latent vectors
def create_random_img( decoder, n = 10, latent_size = 4):
    # Set evaluation mode for the decoder
    decoder.eval()
    random_latent_vectors = []
    with torch.no_grad(): # No need to track the gradients 
        for i in range(n):
            ax = plt.subplot(int(n/5),5,i+1)
            if i==0:
                ax.set_title("created random image by the decoder")
            #get random vector in range (-1,1)
            random_latent_vec  = torch.tensor(-2*np.random.rand(1,latent_size)+1, dtype=torch.float)
            img = decoder(random_latent_vec)
            random_latent_vectors.append(random_latent_vec)
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
        plt.show()

# The function take n image and change every digit in ther latent vector to chack what the impcat of 
# every digit in the vector on the image
def latent_digit_impact(encoder,decoder,n=8, latent_size=4 ,num_of_steps = 8):
    origin_image_list=[]
    image_vec_list =[]
    changed_image_list=[]
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.eval()
    decoder.eval()
    #plo image size
    plt.figure(figsize=(16,4.5))
    #get the image
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    
    with torch.no_grad():
        #save the original image and the latent vector
        for i in range(n):
            #ax = plt.subplot(latent_size,n+3,i+1)
            origin_image_list.append( test_dataset[t_idx[i]][0].unsqueeze(0).to(device))
            image_vec_list.append( encoder(origin_image_list[i]))
        #create the converted image
        for img_index in range(len(image_vec_list)):
            for cordinate in range( latent_size):
                temp_vec = image_vec_list[img_index].detach().clone()
                for step in range(num_of_steps+1):
                    #add the offset to the curent cordinate in the original vector 
                    temp_vec[0][cordinate] = -1 + step * (2 / num_of_steps)
                    changed_image = decoder(temp_vec)
                    #plot the decoder on the original vector
                    if(step==0):
                        #the original image
                        ax = plt.subplot(latent_size,n+3,cordinate*(num_of_steps+3)+1)
                        plt.imshow(origin_image_list[img_index].cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original image")
                        #decoder on the original latent vector
                        ax = plt.subplot(latent_size,n+3,cordinate*(num_of_steps+3)+2)
                        plt.imshow(decoder(image_vec_list[img_index]).cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original vector")
                    #add to the converted image to the plot
                    ax = plt.subplot(latent_size,n+3,cordinate*(num_of_steps+3)+step+3)
                    plt.imshow(changed_image.cpu().squeeze().numpy(), cmap='gist_gray')
                    plt.title(f"cord: {cordinate}. value: {-1 + step * (2 / num_of_steps)}")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False) 
                    #add to the the converted image to the changed list
                    changed_image_list.append(changed_image)
            plt.show()

#save the latent vector in cvs and train liniar logistic on this model
def convert_latent_to_cvs(encoder, latent_size, dataloader,device):
    req_col = {'Latent vector size':[]}
    X = "X"
    for i in range(latent_size):
        temp_col = X+str(i+1)
        req_col[temp_col] = []
    req_col["Y"] = []
    
    #convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    print(df)
    encoder.eval()
    with torch.no_grad(): # No need to track the gradients
        #get the label
        y=test_dataset.targets
        y=y.cpu().detach().numpy()
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)            
            bach_y=y[0:batch_size]
            y=y[batch_size:]
            encoded_data = encoded_data.cpu().detach().numpy()
            rows = np.zeros((batch_size,latent_size+1))
            rows[:,:-1] = encoded_data
            bach_y=bach_y.reshape((-1,1))
            rows[:,-1:] = bach_y
            print("rows\n",rows)
            break



def train_model(lr = 0.001, latent_size = 4 ,num_epochs = 30):

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    #lr= 0.001
    print("lr = ", lr)
    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    #latent_size = 4
    print("latent_size = ", latent_size)
    #model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Encoder(encoded_space_dim=latent_size,fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=latent_size,fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
    
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    #tensorBoard hiper-paramters
    global_step = 1
    #num_epochs = 30
    diz_loss = {'train_loss':[],'val_loss':[]}
    for epoch in range(num_epochs):
       train_loss =train_epoch(encoder,decoder,device,
       train_loader,loss_fn,optim)
       val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
       print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
       diz_loss['train_loss'].append(train_loss)
       diz_loss['val_loss'].append(val_loss)
       temp_name = "Loss, latent size_" +str(latent_size)
       writer.add_scalars(temp_name,{'Traning loss': train_loss,'Test loss':val_loss}, global_step)
       writer.flush()
       global_step +=1
       '''if epoch%10 == 0:
        plot_ae_outputs(encoder,decoder,n=10)'''
    writer.add_scalar( "latent size vs minimun loss", min(diz_loss["val_loss"]), latent_size)
    writer.flush()
    #create_img(decoder,n=4)
    create_random_img( decoder, 10 , latent_size)
    convert_latent_to_cvs(encoder,latent_size, test_loader,device)
    latent_digit_impact(encoder,decoder)
    
    return diz_loss["val_loss"] ,  diz_loss['train_loss']


def latent_size_stat():
    NUM_OF_EPOCH = 2
    STEP_SIZE = 0.25
    NUM_OF_STEPS = 0
    MAX_POW = 5

    ###compare the loss between the train vs test to avoid over fitting
    #creat the cvs colons
    req_col = {'Latent vector size':[]}
    epoch = "Epoch"
    for i in range(NUM_OF_EPOCH):
        temp_col = epoch+" "+str(i+1)
        req_col[temp_col] = []
    req_col["Best epoch"] = []
    req_col["Best loss"] = []
    #convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    #print(df)
    #get the train and test loss for every "2 pow" latent size
    for pow in range(MAX_POW):
        lat_size = int(np.power(2,pow+1))
        test_loss , train_loss = train_model(latent_size = lat_size,num_epochs = NUM_OF_EPOCH)
        
        #add the test loss to data frame 
        row = test_loss 
        for i in range(len(row)):
            row[i]=row[i].item()
        row.insert(0,int(lat_size))
        for i in range(2):
            row.append(None)
        #print("new row = ",row)
        df.loc[len(df)] = row
        print(df)
        

    #find the nim test loss and his epooch and add them to the dataframe
    for i in df.index:
        temp = df.iloc[[i], 1 : NUM_OF_EPOCH+1].values
        print(temp)
        index = np.argmin(temp[0])
        df.iloc[[i],[NUM_OF_EPOCH+1]] = index+1
        df.iloc[[i],[NUM_OF_EPOCH+2]] = temp[0][index]
        print("epoch = ", index+1,", min = ",temp[0][index])
    print(df)
    #save the dataframe as cvs file
    df.to_csv('statistic of latent size and epoch.csv')
    


#latent_size_stat()
train_model(num_epochs = 5)