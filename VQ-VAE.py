import sys
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
#from six.moves import xrange
import os

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import content_enc_builder
from model import dec_builder

from torch.utils.data import Dataset
from PIL import Image, ImageFile
from model.modules import weights_init
import pprint
import json
import tqdm


##############################################################################
# Phase 1: Model Definition - Components used in all phases
##############################################################################


class VectorQuantizer(nn.Module):
    """Basic Vector Quantizer module for VQ-VAE"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)


    def forward(self, inputs):
        """Quantize input vectors using nearest neighbor lookup in embedding space"""
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances between input vectors and codebook embeddings
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) 
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss calculations
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss 

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """Vector Quantizer with Exponential Moving Average updates"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__() 
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon  


    def forward(self, inputs):
        """Quantize inputs with EMA updates to codebook during training"""
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances to codebook vectors
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:  
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Model(nn.Module):
    """Complete VQ-VAE model with encoder, quantizer and decoder"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        self._encoder = content_enc_builder(1,32,256)  # Content encoder
        if decay > 0.0:  
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = dec_builder(32, 1)  # Image decoder

    def forward(self, x):
        """Forward pass through VQ-VAE"""
        z = self._encoder(x)  # [B 256 16 16]
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
    


class CombTrain_VQ_VAE_dataset(Dataset):
    """
    Dataset for VQ-VAE training - loads content font images
    """
    def __init__(self, root, transform = None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # print(img.shape)

    def read_file(self, path):
        """Read all image files from directory"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def __getitem__(self, index):
        """Load and transform single image"""
        img_name = self.imgs[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img) #Tensor [C H W] [1 128 128]
        return img
    
    def __len__(self): 
        return len(self.imgs)


##############################################################################
# Phase 2: Training the VQ-VAE Model
##############################################################################


from torchvision.utils import make_grid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training dataset setup
train_imgs_path = 'path/to/save/train_content_imgs/'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

train_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)

train_loader = DataLoader(train_dataset, batch_size=128, batch_sampler=None, drop_last=True, pin_memory=True, shuffle=True)

# Training parameters
num_training_updates = 50000
embedding_dim = 256
num_embeddings = 100
commitment_cost = 0.25
decay = 0
learning_rate = 2e-4

# Initialize model and optimizer
model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
model.apply(weights_init("xavier"))
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Training loop
model.train()
train_res_recon_error = []
train_res_perplexity = []
train_vq_loss = []

def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

print("Starting VQ-VAE training...")
for i in range(num_training_updates):
    data = next(iter(train_loader))
    train_data_variance = torch.var(data)
    data = data - 0.5 # normalize to [-0.5, 0.5]
    data = data.to(device)
    optimizer.zero_grad()

    # Forward pass
    vq_loss, data_recon, perplexity = model(data)
    # print("vq_loss\n",vq_loss)
    recon_error = F.mse_loss(data_recon, data) / train_data_variance
    loss = recon_error + vq_loss

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Track training metrics
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())
    train_vq_loss.append(vq_loss.item())

    # Log progress
    if (i + 1) % 1000 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-1000:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-1000:]))
        print('vq_loss: %.3f' % np.mean(train_vq_loss[-1000:]))
        print()
        # show(make_grid(data.cpu().data) )


##############################################################################
# Phase 3: Validation/Testing the Trained Model
##############################################################################


print("\nStarting model validation...")

# Validation dataset setup
val_imgs_path = 'path/to/save/val_content_imgs'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
val_dataset = CombTrain_VQ_VAE_dataset(val_imgs_path, transform=tensorize_transform)
validation_loader = DataLoader(val_dataset, batch_size=8, batch_sampler=None, drop_last=True, pin_memory=True, shuffle=True)


def val_(model,validation_loader):
    """Run validation and return original/reconstructed images"""
    model.eval()
    valid_originals = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    # Get model outputs
    vq_output_eval = model._encoder(valid_originals)
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)
    return valid_originals, valid_reconstructions


# Run validation
org, recon_out = val_(model, validation_loader)
# show(make_grid((org+0.5).cpu().data), )
# show(make_grid((recon_out+0.5).cpu().data), )

# Save trained model
print("\nSaving trained model...")
torch.save(model,'/pretrained_weights/VQ-VAE_chn_.pth')    
torch.save(model.state_dict(),'./pretrained_weights/VQ-VAE_Parms_chn_.pth')


##############################################################################
# Phase 4: Calculating Character Similarities Using Trained Encoder
##############################################################################


print("\nCalculating character similarities using trained encoder...")

# Load model for feature extraction
embedding_dim, num_embeddings, commitment_cost, decay = 256, 150, 0.25, 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_embeddings, embedding_dim, commitment_cost, decay).to(device)
models = torch.load('/pretrained_weights/VQ-VAE_chn_.pth')
encoder = models._encoder
encoder.requires_gradq = False

encoder.to("cpu")

class CombTrain_VQ_VAE_dataset(Dataset):
    """Dataset for similarity calculation that preserves image names"""
    def __init__(self, root, transform = None):
        self.img_path = root
        self.transform = transform
        self.imgs = self.read_file(self.img_path)
        # img = Image.open(self.imgs[0])
        # img = self.transform(img)
        # print(img.shape)


    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list


    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img) #Tensor [C H W] [1 128 128]
        ret =(img_name, 
              img
        )
        return ret  # Return both name and image

    def __len__(self):

        return len(self.imgs)


# Setup similarity calculation
train_imgs_path = 'path/to/save/all_content_imgs'
tensorize_transform = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])
batch = 3500 # all content imgs
sim_dataset = CombTrain_VQ_VAE_dataset(train_imgs_path, transform=tensorize_transform)
sim_loader = DataLoader(sim_dataset, batch_size=batch, batch_sampler=None, drop_last=False, pin_memory=True)  
similarity = []


def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


while True:
    data = next(iter(sim_loader)) 
    img_name = data[0]
    img_tensor = data[1]
    img_tensor = img_tensor - 0.5 # normalize to [-0.5, 0.5]
    img_tensor = img_tensor.to("cpu")
    
    # Extract features using trained encoder
    content_feature = encoder(img_tensor)
    # print(content_feature.shape)
    vector = content_feature.view(content_feature.shape[0], -1)
    # print(vector.shape)
    
    # Calculate pairwise similarities
    sim_all = {}
    for i in range(0,batch):  
        char_i = hex(ord(img_name[i][-5]))[2:].upper()  # Extract Unicode from filename
        dict_sim_i = {char_i:{}}
        for j in range(0,batch):
            char_j = hex(ord(img_name[j][-5]))[2:].upper()
            similarity = CosineSimilarity(vector[i],vector[j])
            if i==j:
                similarity=1.0  # Set self-similarity to 1.0
            sim_i2j = {char_j:float(similarity)}
            dict_sim_i[char_i].update(sim_i2j)
        sim_all.update(dict_sim_i)

    # Save similarity matrix to JSON
    dict_json=json.dumps(sim_all) 

    with open('/pretrained_weights/all_char_similarity_unicode.json','w+') as file:
        file.write(dict_json)    
    break


# with open('/data1/chenweiran/SecondPoint/weight50/all_char_similarity_unicode.json','r+') as file:
#     content=file.read()
    
# content=json.loads(content) 
# print(len(content))
# print(content['8774'])