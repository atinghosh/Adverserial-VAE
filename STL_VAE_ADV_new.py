#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os
import pickle
import zipfile
import datetime
from torchvision.models import vgg19, alexnet
num_epochs = 20


# In[2]:

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def collect_output(hook_array):
    output = []
    for hook in hook_array:
        output.append(hook.output.cuda())
    return output

def calculate_percep_loss(input_array, target_array, criterion):
    loss = 0
    assert len(input_array)==len(target_array)
    for i in range(len(input_array)):
        loss += criterion(input_array[i], target_array[i])
    return loss


class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)


class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))


class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128, stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

ndf = 64
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 48 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 24 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 12 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 6 x 6
            nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netD = Discriminator(0).to('cuda')
netD.apply(weights_init)
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(.5, 0.999))
criterion = nn.BCELoss(size_average=False).cuda()
mse_criterion = nn.MSELoss(size_average=False).cuda()


class VAE_ADV(nn.Module):
    def __init__(self, dataset, layer_index = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert dataset in ["mnist", "fashion-mnist", "cifar", "stl"]

        super().__init__()
        # # latent features
        # self.n_latent_features = 64
        self.n_latent_features = 512

        # resolution
        # mnist, fashion-mnist : 28 -> 14 -> 7
        # cifar : 32 -> 8 -> 4
        # stl : 96 -> 24 -> 6
        if dataset in ["mnist", "fashion-mnist"]:
            pooling_kernel = [2, 2]
            encoder_output_size = 7
        elif dataset == "cifar":
            pooling_kernel = [4, 2]
            encoder_output_size = 4
        elif dataset == "stl":
            pooling_kernel = [4, 4]
            encoder_output_size = 6

        # color channels
        if dataset in ["mnist", "fashion-mnist"]:
            color_channels = 1
        else:
            color_channels = 3

        # # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)

        # data
        self.train_loader, self.test_loader = self.load_data(dataset)
        # history
        self.history = {"loss": [], "val_loss": []}

        # model name
        self.model_name = dataset
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

        self.alexnet = alexnet(pretrained=True).cuda()
        self.alexnet.eval()
        for param in self.alexnet.parameters():
            param.requires_grad = False
        if layer_index is not None:
            layers = [self.alexnet.features[ind] for ind in layer_index]  # layers for which we need the output
            self.hookF = [Hook(layer) for layer in layers]

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(64, self.n_latent_features).to(self.device)
        z = self.fc3(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc3(z)
        d = self.decoder(z)
        return d, mu, logvar

    def load_data(self, dataset):
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if dataset == "mnist":
            train = MNIST(root="./data", train=True, transform=data_transform, download=True)
            test = MNIST(root="./data", train=False, transform=data_transform, download=True)
        elif dataset == "fashion-mnist":
            train = FashionMNIST(root="./data", train=True, transform=data_transform, download=True)
            test = FashionMNIST(root="./data", train=False, transform=data_transform, download=True)
        elif dataset == "cifar":
            train = CIFAR10(root="./data", train=True, transform=data_transform, download=True)
            test = CIFAR10(root="./data", train=False, transform=data_transform, download=True)
        elif dataset == "stl":
            train = STL10(root="/data02/Atin/STL10/", split="unlabeled", transform=data_transform, download=True)
            test = STL10(root="/data02/Atin/STL10/", split="test", transform=data_transform, download=True)

        train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, num_workers=0)

        return train_loader, test_loader

    def loss_function(self, recon_x, x, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE, KLD

    def init_model(self):

        # self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002, betas=(.5, 0.999))


        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark = True
        self.to(self.device)
        self.apply(weights_init)

    # Train
    def fit_adv_train(self, epoch):
        self.train()
        print(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        train_vae_loss = 0
        train_adv_loss = 0
        train_percep_loss = 0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)

            netD.zero_grad()
            b_size = inputs.size(0)
            label = torch.full((b_size,), real_label, device=self.device)
            # Forward pass real batch through D
            output = netD(inputs).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            recon_batch, mu, logvar = self(inputs)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(recon_batch.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            self.zero_grad()
            label.fill_(real_label)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(recon_batch).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)
            BCE, KLD = self.loss_function(recon_batch, inputs, mu, logvar)
            vae_loss = BCE + 10 * KLD

            # Perceptual loss
            _ = self.alexnet(inputs)
            input_features = collect_output(self.hookF)
            _ = self.alexnet(recon_batch)
            output_features = collect_output(self.hookF)
            perceptual_loss = calculate_percep_loss(output_features, input_features, mse_criterion)




            loss = 100 * errG + BCE + 10 * KLD + 1000 * perceptual_loss

            loss.backward()
            D_G_z2 = output.mean().item()
            self.optimizer.step()

            train_loss += loss.item()
            train_vae_loss += vae_loss.item()
            train_adv_loss += errG.item()
            train_percep_loss += perceptual_loss.item()
            samples_cnt += inputs.size(0)


            if batch_idx % 50 == 0:
                print(batch_idx, len(self.train_loader), f"Loss: {train_loss / samples_cnt:f}")
                print(batch_idx, len(self.train_loader), f"VAE_Loss: {train_vae_loss / samples_cnt:f}")
                print(batch_idx, len(self.train_loader), f"ADV_Loss: {train_adv_loss / samples_cnt:f}")
                print(batch_idx, len(self.train_loader), f"Perceptual_Loss: {train_percep_loss / samples_cnt:f}")

                print("FEW MORE STAT......")
                print(f"adv loss is {errG.item()}")
                print(f"recon loss is {BCE.item()}")
                print(f"KLD loss is {KLD.item()}")
                print(f"perceptual loss is {perceptual_loss.item()}")

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, batch_idx, len(self.train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        self.history["loss"].append(train_loss / samples_cnt)


    # Train
    def fit_train(self, epoch):
        self.train()
        print(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(inputs)

            loss = self.loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            samples_cnt += inputs.size(0)

            if batch_idx % 50 == 0:
                print(batch_idx, len(self.train_loader), f"Loss: {train_loss / samples_cnt:f}")

        self.history["loss"].append(train_loss / samples_cnt)

    def test_adv(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                recon_batch, mu, logvar = self(inputs)
                bce, kld = self.loss_function(recon_batch, inputs, mu, logvar)
                vae_loss = bce + kld
                val_loss += vae_loss.item()
                samples_cnt += inputs.size(0)

                if batch_idx == 0:
                    save_image(recon_batch, f"{self.model_name}/reconstruction_epoch_{str(epoch)}.png", nrow=8)

        print(batch_idx, len(self.test_loader), f"ValLoss: {val_loss / samples_cnt:f}")
        self.history["val_loss"].append(val_loss / samples_cnt)

        save_image(self.sampling(), f"{self.model_name}/sampling_epoch_{str(epoch)}.png", nrow=8)

    # def test(self, epoch):
    #     self.eval()
    #     val_loss = 0
    #     samples_cnt = 0
    #     with torch.no_grad():
    #         for batch_idx, (inputs, _) in enumerate(self.test_loader):
    #             inputs = inputs.to(self.device)
    #             recon_batch, mu, logvar = self(inputs)
    #             val_loss += self.loss_function(recon_batch, inputs, mu, logvar).item()
    #             samples_cnt += inputs.size(0)
    #
    #             if batch_idx == 0:
    #                 save_image(recon_batch, f"{self.model_name}/reconstruction_epoch_{str(epoch)}.png", nrow=8)
    #
    #     print(batch_idx, len(self.test_loader), f"ValLoss: {val_loss / samples_cnt:f}")
    #     self.history["val_loss"].append(val_loss / samples_cnt)
    #
    #     save_image(self.sampling(), f"{self.model_name}/sampling_epoch_{str(epoch)}.png", nrow=8)

    # save results
    def save_history(self):
        with open(f"{self.model_name}/{self.model_name}_history.dat", "wb") as fp:
            pickle.dump(self.history, fp)

    def save_to_zip(self):
        with zipfile.ZipFile(f"{self.model_name}.zip", "w") as zip:
            for file in os.listdir(self.model_name):
                zip.write(f"{self.model_name}/{file}", file)




# In[3]:


def main():
    net = VAE_ADV("stl", layer_index=[6])
    net.init_model()
    for i in range(num_epochs):
        net.fit_adv_train(i)
        net.test_adv(i)
        net.save_history()
        net.save_to_zip()


# In[4]:


main()

# In[ ]:




