import matplotlib.pyplot as plt
import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm
import os
import utils



use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    

class MNISTEncoder(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(MNISTEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        

    def forward(self, x):
        """x = sparsely sampled image
        this returns the aggregated r value
        """

        output = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))

        return output.mean(0).view(1, 128)

class MNISTEncoderWithAttention(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(MNISTEncoderWithAttention, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        

    def forward(self, x):
        """x = sparsely sampled image
        this returns the aggregated r value
        """

        output = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        return F.softmax(torch.mm(torch.mm(output, torch.transpose(output, 0,1)) / np.sqrt(output.shape[1]), output), dim=0).mean(0).view(1, 128)

class MNISTLatentEncoder(nn.Module):
    """takes in context points and returns a fixed length aggregation"""
    def __init__(self):
        super(MNISTLatentEncoder, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 128)
        self.mu_latent = nn.Linear(128,1)
        self.sigma_latent = nn.Linear(128,1)
        

    def forward(self, x):
        """x = sparsely sampled image
        this returns the aggregated r value
        """

        output = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))

        hidden = output.mean(0).view(1, 128)

        #first pass through intermediate relu layer
        hidden = F.relu(self.fc5(hidden))

        #then pass through these other layers
        mu = self.mu_latent(hidden)
        sigma = self.sigma_latent(hidden)

        sigma = 0.1 + 0.9*F.softplus(sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class MNISTDecoder(nn.Module):
    def __init__(self, m, n):
        super(MNISTDecoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(130, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, out):
        """r is the aggregated data used to condition, out is the concatenation with the frame"""        

        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(out)))))))
        return h
        mu = h[:,0]
        log_sigma = h[:,1]

        # bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        return mu, sigma

class MNISTLatentDecoder(nn.Module):
    def __init__(self, m, n):
        super(MNISTLatentDecoder, self).__init__()
        self.m = m
        self.n = n
        self.fc1 = nn.Linear(131, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
    def forward(self, out):
        """r is the aggregated data used to condition, out is the concatenation with the frame"""        

        h = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(out)))))))
        # return h
        mu = h[:,0]
        log_sigma = h[:,1]

        # bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        return torch.distributions.Normal(loc=mu, scale=sigma), mu, sigma


class FCCritic(nn.Module):
    def __init__(self):
        super(FCCritic, self).__init__()
        self.fc1 = nn.Linear(28*28, 2*28*28) 
        self.fc2 = nn.Linear(2*28*28, 3*28*28) 
        self.fc3 =nn.Linear(3*28*28, 2*28*28) 
        self.fc4 =nn.Linear(2*28*28, 1*28*28) 
        self.output = nn.Linear(1*28*28, 1)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))
        out = self.output(out)
        return out.view(-1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 2*64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(2*64, 4*64, kernel_size=5, stride=2, padding=2)
        self.output = nn.Linear(4*4*4*64, 1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        out = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
        out = out.view(-1, 4*4*4*64)
        out = self.output(out)
        return out.view(-1)

if __name__ == "__main__":

    use_attention = False
    batch_size= 1
    test_batch_size=1

    m, n = 28, 28
    num_pixels = m*n

    min_context_points = num_pixels * 0.15 # always have at least 15% of all pixels
    max_context_points = num_pixels * 0.95 # always have at most 95% of all pixels


    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    # kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    epochs = 10
    log_interval = 50
    learning_rate = 0.1
    n_critic = 5
    
    if use_attention:
        encoder = MNISTEncoderWithAttention().to(device)
    else:
        encoder = MNISTEncoder().to(device)
    latent_encoder = MNISTLatentEncoder().to(device)
    decoder = MNISTLatentDecoder(m, n).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(latent_encoder.parameters()), lr=0.0001, betas=(0,0.9))
    
    image_frame = torch.tensor([[i, j] for i in range(0,m) for j in range(0,n)]).float().to(device)
    
    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()

        progress = tqdm(train_loader)

        total_loss = 0
        count = 0
        # for (batch_ground_truth_image, target) in progress:
            # ground_truth_image = ground_truth_image.view(28, 28).to(device)
            # loss = 0

        for (ground_truth_image, target) in progress:
            ground_truth_image = ground_truth_image.view(28, 28).to(device)

            sparse_data = utils.get_mnist_context_points(ground_truth_image, context_points=np.random.randint(min_context_points, max_context_points)).float().to(device)

            data = utils.get_mnist_features(sparse_data)

            data = data.to(device).float()
            
            # run the model to get r which will be concatenated onto every node pair in the decoder
            r = encoder(data)
            posterior = latent_encoder(data)
            latent_rep = posterior.sample()
            # latent_rep = latent_rep.view(1,-1).repeat(1,m*n).view(m*n, 128) 
                            
            r = torch.cat((r, latent_rep), -1)
            # print(r.shape)
            out = torch.cat((image_frame, r.view(1,-1).repeat(1,m*n).view(m*n,129)), 1)
            # out = torch.cat((image_frame, r), -1) 

            dist, mu, sigma = decoder(out)
            mu = mu.view(m,n)
            sigma = sigma.view(m,n)
            temp_loss = utils.get_log_p(ground_truth_image, mu, sigma).mean() - torch.distributions.kl_divergence(dist, posterior).mean()

            loss = -temp_loss 
            # loss = loss / batch_size
            # total_loss += loss.item() 

            # count += 1
            
            loss.backward()
            optimizer.step()
            progress.set_description('E:{} - Loss: {:.4f}'.format(epoch, loss.item()))

        with open("encoder_wasnp.pkl", "wb") as of:
             pickle.dump(encoder, of)

        with open("decoder_wasnp.pkl", "wb") as of:
             pickle.dump(decoder, of)

        with open("optim.pkl", "wb") as of:
             pickle.dump(optimizer, of)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():

            for i, (ground_truth_image, target) in enumerate(test_loader):
                ground_truth_image = ground_truth_image.view(28, 28).to(device)

                sparse_data = utils.get_mnist_context_points(ground_truth_image, context_points=400).float().to(device)

                data = utils.get_mnist_features(sparse_data)
                data = data.to(device).float()

                r = encoder(data)
                posterior = latent_encoder(data)
                latent_rep = posterior.sample()
                             
                r = torch.cat((r, latent_rep), -1)
                out = torch.cat((image_frame, r.view(1,-1).repeat(1,m*n).view(m*n,129)), 1)

                dist, mu, sigma = decoder(out)
                mu = mu.view(m,n)
                sigma = sigma.view(m,n)


                fig, ax = plt.subplots(ncols=2, nrows=2)
                ax[0][1].imshow(ground_truth_image.reshape(m,n).cpu())
                ax[0][0].imshow(sparse_data.reshape(m,n).cpu())
                ax[1][0].imshow(mu.detach().reshape(m,n).cpu())
                ax[1][1].imshow(sigma.detach().reshape(m,n).cpu())
                plt.savefig("attention{}res{}.png".format(epoch, i))
                plt.close()
                # plt.imsave("{}context_points{}.png".format(epoch, i), sparse_data.reshape(m,n).cpu(), dpi=300)

                # plt.imsave("{}ground_truth{}.png".format(epoch, i), ground_truth_image.reshape(m,n).cpu() ,dpi=300)

                # plt.imsave("{}distr{}.png".format(epoch, i), h.detach().reshape(m,n).cpu(), dpi=300)
                if i >= 10:
                    break
