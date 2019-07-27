import numpy as np
import math
import copy

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal

# Torchvision
import torchvision
import torchvision.transforms as transforms

# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

Lr = .001
Nb_epoch = 70
Epochs = Nb_epoch
Lr_rampdown_epochs = 1.3 * Nb_epoch
NOISE_RATIO = .4


def noise_input(images):
    return images * (1 - NOISE_RATIO) + torch.rand_like(images) * NOISE_RATIO


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = copy.deepcopy(Lr)
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    #     lr = linear_rampup(epoch, Lr_rampup) * (Lr - Initial_lr) + Initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if Lr_rampdown_epochs:
        assert Lr_rampdown_epochs >= Epochs
        lr *= cosine_rampdown(epoch, Lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             nn.ReLU(),
# # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
# #             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
# #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
# #             nn.ReLU(),
# 			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

# really small dimensional autoencoder representation
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             nn.ReLU(),
# 			# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             # nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             # nn.ReLU(),
# 			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             nn.ReLU(),
# 			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
#             nn.ReLU(),
#             nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         encoded_res = encoded.size()[1:]
#
#         encoded_mod = encoded.view(b_size, -1)
#         encoded_mod_dim = encoded_mod.size()[1]
#         encoded_mod = encoded_mod / torch.norm(encoded_mod, dim=1, keepdim=True)
#         encoded_mod = encoded_mod * math.sqrt(encoded_mod_dim)
#         encoded_mod = encoded_mod.view(-1, *encoded_res)
#
#         decoded = self.decoder(encoded_mod)
#         return encoded, encoded_mod, decoded


# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
# 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
#             # nn.BatchNorm2d(48),
#             nn.ReLU(),
# 			# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             # nn.ReLU(),
#         )
#         self.linear_encoder = nn.Linear(48*4*4, 128)
#         self.linear_decoder = nn.Linear(128, 48*4*4)
#         self.decoder = nn.Sequential(
#             # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
# 			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
# 			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         encoded = self.linear_encoder(encoded.view(b_size, -1))
#         noisy_encoded = encoded + .2 * torch.randn_like(encoded)
#
#         decoded = self.linear_decoder(noisy_encoded).view(-1, 48, 4, 4)
#         decoded = self.decoder(decoded)
#
#         return encoded, decoded


# encoded_res = encoded.size()[1:]
#
# encoded_mod = encoded.view(b_size, -1)
# encoded_mod_dim = encoded_mod.size()[1]
# encoded_mod = encoded_mod / torch.norm(encoded_mod, dim=1, keepdim=True)
# encoded_mod = encoded_mod * math.sqrt(encoded_mod_dim)
# encoded_mod = encoded_mod.view(-1, *encoded_res)
#
# decoded = self.decoder(encoded_mod)
# return encoded, encoded_mod, decoded


# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         k_size = 5
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 48, k_size, stride=2, padding=2),  # [batch, 24, 48, 48]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Conv2d(48, 96, k_size, stride=2, padding=2),  # [batch, 48, 24, 24]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.Conv2d(96, 192, k_size, stride=2, padding=2),  # [batch, 96, 12, 12]
#             # nn.BatchNorm2d(48),
#             nn.ReLU(),
#             nn.Conv2d(192, 384, k_size, stride=2, padding=2),  # [batch, 192, 6, 6]
#             nn.ReLU(),
#             nn.Conv2d(384, 768, k_size, stride=2, padding=2),  # [batch, 768, 3, 3]
#             nn.ReLU(),
#         )
#         self.linear_encoder = nn.Linear(768 * 3 * 3, 1536)
#         self.linear_decoder = nn.Linear(1536, 768 * 3 * 3)
#         self.decoder = nn.Sequential(
#             nn.ReLU(),
#             nn.ConvTranspose2d(768, 384, 5, stride=2, padding=2, output_padding=1),  # [batch, 48, 6, 6]
#             nn.ReLU(),
#             nn.ConvTranspose2d(384, 192, 5, stride=2, padding=2, output_padding=1),  # [batch, 24, 16, 16]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.ConvTranspose2d(192, 96, 5, stride=2, padding=2, output_padding=1),  # [batch, 12, 32, 32]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.ReLU(),
#             nn.ConvTranspose2d(48, 3, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         encoded = self.linear_encoder(encoded.view(b_size, -1))
#         noisy_encoded = encoded + .2 * torch.randn_like(encoded)
#
#         decoded = self.linear_decoder(noisy_encoded).view(-1, 768, 3, 3)
#         decoded = self.decoder(decoded)
#
#         return encoded, decoded


# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         k_size = 5
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 48, k_size, stride=2, padding=2),  # [batch, 24, 48, 48]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Conv2d(48, 96, k_size, stride=2, padding=2),  # [batch, 48, 24, 24]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.Conv2d(96, 192, k_size, stride=2, padding=2),  # [batch, 96, 12, 12]
#             # nn.BatchNorm2d(48),
#             nn.ReLU(),
#             nn.Conv2d(192, 384, k_size, stride=2, padding=2),  # [batch, 192, 6, 6]
#             # nn.ReLU(),
#             # nn.Conv2d(384, 768, k_size, stride=2, padding=2),  # [batch, 768, 3, 3]
#             # nn.ReLU(),
#         )
#         # self.linear_encoder = nn.Linear(768 * 3 * 3, 1536)
#         # self.linear_decoder = nn.Linear(1536, 768 * 3 * 3)
#         self.decoder = nn.Sequential(
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(768, 384, 5, stride=2, padding=2, output_padding=1),  # [batch, 48, 6, 6]
#             nn.ReLU(),
#             nn.ConvTranspose2d(384, 192, 5, stride=2, padding=2, output_padding=1),  # [batch, 24, 16, 16]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.ConvTranspose2d(192, 96, 5, stride=2, padding=2, output_padding=1),  # [batch, 12, 32, 32]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.ReLU(),
#             nn.ConvTranspose2d(48, 3, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         # encoded = self.linear_encoder(encoded.view(b_size, -1))
#         noisy_encoded = encoded + .2 * torch.randn_like(encoded)
#
#         # decoded = self.linear_decoder(noisy_encoded).view(-1, 768, 3, 3)
#         # decoded = self.decoder(decoded)
#         decoded = self.decoder(noisy_encoded)
#
#         return encoded, decoded


# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         k_size = 5
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 24, k_size, stride=2, padding=2),  # [batch, 24, 48, 48]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Conv2d(24, 48, k_size, stride=2, padding=2),  # [batch, 48, 24, 24]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.Conv2d(48, 96, k_size, stride=2, padding=2),  # [batch, 96, 12, 12]
#             # nn.BatchNorm2d(48),
#             nn.ReLU(),
#             nn.Conv2d(96, 192, k_size, stride=2, padding=2),  # [batch, 192, 6, 6]
#             # nn.ReLU(),
#             # nn.Conv2d(384, 768, k_size, stride=2, padding=2),  # [batch, 768, 3, 3]
#             # nn.ReLU(),
#         )
#         # self.linear_encoder = nn.Linear(768 * 3 * 3, 1536)
#         # self.linear_decoder = nn.Linear(1536, 768 * 3 * 3)
#         self.decoder = nn.Sequential(
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(768, 384, 5, stride=2, padding=2, output_padding=1),  # [batch, 48, 6, 6]
#             nn.ReLU(),
#             nn.ConvTranspose2d(192, 96, 5, stride=2, padding=2, output_padding=1),  # [batch, 24, 16, 16]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # [batch, 12, 32, 32]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.ConvTranspose2d(48, 24, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 3, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         # encoded = self.linear_encoder(encoded.view(b_size, -1))
#         noisy_encoded = encoded + .2 * torch.randn_like(encoded)
#
#         # decoded = self.linear_decoder(noisy_encoded).view(-1, 768, 3, 3)
#         # decoded = self.decoder(decoded)
#         decoded = self.decoder(noisy_encoded)
#
#         return encoded, decoded


# class Autoencoder(nn.Module):
#
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         k_size = 5
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 12, k_size, stride=2, padding=2),  # [batch, 24, 48, 48]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.Conv2d(12, 24, k_size, stride=2, padding=2),  # [batch, 48, 24, 24]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.Conv2d(24, 48, k_size, stride=2, padding=2),  # [batch, 96, 12, 12]
#             # nn.BatchNorm2d(48),
#             nn.ReLU(),
#             nn.Conv2d(48, 96, k_size, stride=2, padding=2),  # [batch, 192, 6, 6]
#             # nn.ReLU(),
#             # nn.Conv2d(384, 768, k_size, stride=2, padding=2),  # [batch, 768, 3, 3]
#             # nn.ReLU(),
#         )
#         # self.linear_encoder = nn.Linear(768 * 3 * 3, 1536)
#         # self.linear_decoder = nn.Linear(1536, 768 * 3 * 3)
#         self.decoder = nn.Sequential(
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(768, 384, 5, stride=2, padding=2, output_padding=1),  # [batch, 48, 6, 6]
#             nn.ReLU(),
#             nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # [batch, 24, 16, 16]
#             # nn.BatchNorm2d(24),
#             nn.ReLU(),
#             nn.ConvTranspose2d(48, 24, 5, stride=2, padding=2, output_padding=1),  # [batch, 12, 32, 32]
#             # nn.BatchNorm2d(12),
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 12, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.ReLU(),
#             nn.ConvTranspose2d(12, 3, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b_size = x.size()[0]
#
#         encoded = self.encoder(x)
#         # encoded = self.linear_encoder(encoded.view(b_size, -1))
#         noisy_encoded = encoded + .2 * torch.randn_like(encoded)
#
#         # decoded = self.linear_decoder(noisy_encoded).view(-1, 768, 3, 3)
#         # decoded = self.decoder(decoded)
#         decoded = self.decoder(noisy_encoded)
#
#         return encoded, decoded



class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        k_size = 5
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, k_size, stride=2, padding=2),  # [batch, 24, 32, 32]
            # nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(24, 48, k_size, stride=2, padding=2),  # [batch, 48, 16, 16]
            # nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(48, 96, k_size, stride=2, padding=2),  # [batch, 96, 8, 8]
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(96, 96 * 2, k_size, stride=2, padding=2),  # [batch, 192, 4, 4]
            nn.ReLU(),
        )
        self.linear_encoder = nn.Linear(192 * 4 * 4, 128)
        self.linear_decoder = nn.Linear(128, 192 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 5, stride=2, padding=2, output_padding=1),  # [batch, 48, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1),  # [batch, 24, 16, 16]
            # nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 5, stride=2, padding=2, output_padding=1),  # [batch, 12, 32, 32]
            # nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 5, stride=2, padding=2, output_padding=1),  # [batch, 3, 64, 64]
            nn.Sigmoid(),
        )

    def forward(self, x):
        b_size = x.size()[0]

        encoded = self.encoder(x)
        encoded = self.linear_encoder(encoded.view(b_size, -1))
        noisy_encoded = encoded + .2 * torch.randn_like(encoded)

        decoded = self.linear_decoder(noisy_encoded).view(-1, 192, 4, 4)
        decoded = self.decoder(decoded)

        return encoded, decoded




def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()

    # Load data
    transform = transforms.Compose([transforms.ToTensor(), ])
    transform = transforms.Compose(
        [ transforms.Resize((64,64)),
        transforms.RandomAffine(degrees=10, translate=(.03, .03), shear=3),
         # transforms.ColorJitter(brightness=2),
         transforms.ToTensor(), ])
    trainset = torchvision.datasets.STL10(root='/data02/Atin/STL10/', split="train",
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=False, num_workers=10)




    # if args.valid:
    #     print("Loading checkpoint...")
    #     autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
    #     dataiter = iter(testloader)
    #     images, labels = dataiter.next()
    #     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
    #     imshow(torchvision.utils.make_grid(images))
    #
    #     images = Variable(images.cuda())
    #
    #     decoded_imgs = autoencoder(images)[1]
    #     imshow(torchvision.utils.make_grid(decoded_imgs.data))
    #
    #     exit(0)

    # Define an optimizer and criterion
    criterion = nn.BCELoss().cuda()
    # criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=Lr, weight_decay=.0001)
    optimizer = optim.Adam(autoencoder.parameters(), lr=Lr)

    # optimizer = torch.optim.SGD(autoencoder.parameters(), Lr,
    #                             momentum=.9,
    #                             weight_decay=.0001,
    #                             nesterov=False)

    for epoch in range(Nb_epoch):
        running_loss = 0.0
        count = 0
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded, outputs = autoencoder(inputs)
            # encoded = encoded.view(inputs.size()[0], -1)

            # encoded_norm_loss = torch.mean(torch.norm(encoded, dim=1)**2)
            loss = criterion(outputs, inputs)
            # final_loss = loss + .5 * encoded_norm_loss
            # ============ Backward ============
            adjust_learning_rate(optimizer, epoch, i, len(trainloader) + 1)
            optimizer.zero_grad()
            loss.backward()
            # final_loss.backward()
            optimizer.step()

            # ============ Logging ============
            running_loss += loss.data
            count += 1
            if i % 200 == 0:
                print('Step: {}, Epoch: {} and loss: {:05.3f}'.format(i, epoch + 1, loss.item()))
                print("running loss is: {:05.3f}".format(running_loss / count))
                running_loss = 0.0
                count = 0
                # print(loss.item())
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 100))
                # running_loss = 0.0

    print('Finished Training')
    print('Saving Model...')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    torch.save(autoencoder.state_dict(), "./weights/autoencoder_STL3_fully_conv.pkl")


# def main():
#     parser = argparse.ArgumentParser(description="Train Autoencoder")
#     parser.add_argument("--valid", action="store_true", default=False,
#                         help="Perform validation only.")
#     args = parser.parse_args()
#
#     # Create model
#     autoencoder = create_model()
#
#     # Load data
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(), ])
#     trainset = torchvision.datasets.CIFAR10(root='/data02/Atin/CIFAR10/', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                               shuffle=True, num_workers=2)
#     testset = torchvision.datasets.CIFAR10(root='/data02/Atin/CIFAR10/', train=False,
#                                            download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=16,
#                                              shuffle=False, num_workers=2)
#     classes = ('plane', 'car', 'bird', 'cat',
#                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     # if args.valid:
#     #     print("Loading checkpoint...")
#     #     autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
#     #     dataiter = iter(testloader)
#     #     images, labels = dataiter.next()
#     #     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
#     #     imshow(torchvision.utils.make_grid(images))
#     #
#     #     images = Variable(images.cuda())
#     #
#     #     decoded_imgs = autoencoder(images)[1]
#     #     imshow(torchvision.utils.make_grid(decoded_imgs.data))
#     #
#     #     exit(0)
#
#     # Define an optimizer and criterion
#     criterion = nn.BCELoss().cuda()
#     # criterion = nn.MSELoss().cuda()
#     optimizer = optim.Adam(autoencoder.parameters(), lr=Lr, weight_decay=.0001)
#
#     # optimizer = torch.optim.SGD(autoencoder.parameters(), Lr,
#     #                             momentum=.9,
#     #                             weight_decay=.0001,
#     #                             nesterov=False)
#
#     for epoch in range(100):
#         running_loss = 0.0
#         total_running_loss = 0.0
#         count = 0
#         for i, (inputs, _) in enumerate(trainloader, 0):
#             inputs = get_torch_vars(inputs)
#
#             # ============ Forward ============
#             encoded, outputs = autoencoder(inputs)
#             encoded = encoded.view(inputs.size()[0], -1)
#
#             encoded_norm_loss = torch.mean(torch.norm(encoded, dim=1)**2)
#             loss = criterion(outputs, inputs)
#             final_loss = loss + .5 * .5 * encoded_norm_loss
#             # ============ Backward ============
#             adjust_learning_rate(optimizer, epoch, i, len(trainloader) + 1)
#             optimizer.zero_grad()
#             # loss.backward()
#             final_loss.backward()
#             optimizer.step()
#
#             # ============ Logging ============
#             running_loss += loss.data
#             total_running_loss +=final_loss.data
#             count += 1
#             if i % 100 == 0:
#                 print('Step: {}, Epoch: {} and loss: {:05.3f}'.format(i, epoch+1, loss.item()))
#                 print("running loss is: {:05.3f}".format(running_loss/count))
#                 print("Total loss is: {:05.3f}".format(total_running_loss/count))
#                 running_loss = 0.0
#                 total_running_loss = 0.0
#                 count = 0
#                 # print(loss.item())
#                 # print('[%d, %5d] loss: %.3f' %
#                 #       (epoch + 1, i + 1, running_loss / 100))
#                 # running_loss = 0.0
#
#     print('Finished Training')
#     print('Saving Model...')
#     if not os.path.exists('./weights'):
#         os.mkdir('./weights')
#     torch.save(autoencoder.state_dict(), "./weights/autoencoder_latest_that_works_final_ADAM_latest_RAE_768.pkl")
#     print("RAE with 768 dimensional latent space")

if __name__ == '__main__':
    main()
