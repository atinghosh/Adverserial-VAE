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
from torchvision.models import vgg19

# OS
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def calculate_kl(mu, logvar, reduce_mean = True):
    sigma = torch.exp(.5 * logvar)
    single_term = torch.norm(mu,dim=1)**2 + torch.norm(sigma,dim=1)**2 - torch.sum(logvar, 1)
    if reduce_mean:
        return .5 * torch.mean(single_term)
    else:
        return .5 * torch.sum(single_term)

def size_cal(input_size, kernel_size, padding, stride):
    x = (input_size + 2 * padding - kernel_size)/stride
    return math.floor(x + 1)

class Discriminator(nn.Module):

    def __init__(self, p):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            # nn.Conv2d(3, 32, 7, stride=2, padding=0),  # [batch, 32, 64, ],
            # nn.ReLU,
            nn.Conv2d(3, 32, 7, stride=2, padding=0),  # [batch, 24, 32, 32]
            # nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1, padding=0),  # [batch, 48, 16, 16]
            # nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),  # [batch, 96, 8, 8]
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=0),  # [batch, 192, 4, 4]
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=0),  # [batch, 192, 4, 4]
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.drop_layer1 = nn.Dropout(p=p)
        self.fc1 = nn.Linear(256, 128)
        self.drop_layer2 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.drop_layer1(self.global_avg_pool(x))
        x = x.view(-1, 256)
        x = self.drop_layer2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = F.sigmoid(x) + 1e-8
        return x


def train_discriminator(autoencoder_model, discriminator_model, opt, input):

    with torch.no_grad():
        _, _, output = autoencoder_model(input)
    discriminator_model.train()
    loss = -torch.log(discriminator_model(input)) - torch.log(1 - discriminator_model(output))
    loss = torch.mean(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 64, 64]
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
        self.linear_encoder_mu = nn.Linear(192 * 6 * 6, 1024)
        self.linear_encoder_log_var = nn.Linear(192 * 6 * 6, 1024)

        self.linear_decoder = nn.Linear(1024, 192 * 6 * 6)
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
        encoded_mu = self.linear_encoder_mu(encoded.view(b_size, -1))
        encoded_log_var = self.linear_encoder_log_var(encoded.view(b_size, -1))

        std = torch.exp(.5 * encoded_log_var)
        noisy_encoded = encoded_mu + torch.randn_like(encoded_mu) * std

        decoded = self.linear_decoder(noisy_encoded).view(-1, 192, 6, 6)
        decoded = self.decoder(decoded)

        return encoded_mu, encoded_log_var, decoded


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



def main():

    def collect_output(hook_array):
        output = []
        for hook in hook_array:
            output.append(hook.output.cuda())
        return output

    def calculate_loss(input_array, target_array, criterion):
        loss = 0
        assert len(input_array)==len(target_array)
        for i in range(len(input_array)):
            loss += criterion(input_array[i], target_array[i])
        return .5 * loss


    # Create model
    autoencoder = create_model()
    vgg_model = vgg19(pretrained=True).cuda()
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False

    discriminator_model = Discriminator(.5).cuda()
    optimizer_discriminator = optim.Adam(discriminator_model.parameters(), lr=.0001)

    # Load data
    # transform = transforms.Compose([transforms.ToTensor(), ])
    transform = transforms.Compose(
        [
            # transforms.Resize((64,64)),
            transforms.RandomAffine(degrees=10, translate=(.03, .03), shear=3),
         # transforms.ColorJitter(brightness=2),
         transforms.ToTensor(), ])
    trainset = torchvision.datasets.STL10(root='/data02/Atin/STL10', split="train",
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
    mse_criterion = nn.MSELoss().cuda()

    # criterion = nn.MSELoss().cuda()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=Lr, weight_decay=.0001)
    optimizer = optim.Adam(autoencoder.parameters(), lr=Lr)

    # optimizer = torch.optim.SGD(autoencoder.parameters(), Lr,
    #                             momentum=.9,
    #                             weight_decay=.0001,
    #                             nesterov=False)

    # l_index = [3, 8, 11, 15, ]
    # l_index = [2, 5, 7, 10, 14, 19, 28]
    # l_index = [5, 7, 10]
    l_index = [10, 14, 19]





    layers = [vgg_model.features[ind] for ind in l_index]  # layers for which we need the output
    hookF = [Hook(layer) for layer in layers]

    for epoch in range(Nb_epoch):
        running_loss = 0.0
        count = 0
        for i, (inputs, _) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)

            # ============ Forward ============
            encoded_mu, encoded_sigma, outputs = autoencoder(inputs)

            # Training the discriminator
            train_discriminator(autoencoder, discriminator_model, optimizer_discriminator, inputs)

            # encoded = encoded.view(inputs.size()[0], -1)
            # inputs = F.interpolate(inputs, (224, 224), mode="bilinear")
            # outputs = F.interpolate(outputs, (224, 224), mode="bilinear")

            _ = vgg_model(inputs)
            input_vgg_features = collect_output(hookF)
            _ = vgg_model(outputs)
            output_vgg_features = collect_output(hookF)

            # bce_loss = criterion(outputs, inputs)
            bce_loss = mse_criterion(outputs, inputs)

            perceptual_loss = calculate_loss(output_vgg_features, input_vgg_features, mse_criterion)
            kl_loss = calculate_kl(encoded_mu, encoded_sigma)
            # final_loss = perceptual_loss +  .5 * .5 * encoded_norm_loss
            with torch.no_grad():
                discriminator_model.eval()
                discriminator_output = discriminator_model(outputs)
            adverserial_loss = -torch.log(discriminator_output)
            adverserial_loss = torch.mean(adverserial_loss)


            final_loss = bce_loss + 0 * perceptual_loss + .001 * kl_loss + .1 * adverserial_loss
            adjust_learning_rate(optimizer, epoch, i, len(trainloader) + 1)
            optimizer.zero_grad()
            # perceptual_loss.backward()
            final_loss.backward()
            optimizer.step()


            # ============ Logging ============
            running_loss += perceptual_loss.data
            count += 1
            if i % 200 == 0:
                print('Step: {}, Epoch: {} and percep_loss: {:05.3f}'.format(i, epoch + 1, perceptual_loss.item()))
                print('Step: {}, Epoch: {} and total_loss: {:05.3f}'.format(i, epoch + 1, final_loss.item()))
                print('Step: {}, Epoch: {} and bce_loss: {:05.3f}'.format(i, epoch + 1, bce_loss.item()))
                print('Step: {}, Epoch: {} and kl_loss: {:05.3f}'.format(i, epoch + 1, kl_loss.item()))
                print('Step: {}, Epoch: {} and adverserial loss: {:05.3f}'.format(i, epoch + 1, adverserial_loss.item()))
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
        # torch.save(autoencoder.state_dict(), "./weights/autoencoder_percept_STL3.pkl")
        torch.save(autoencoder.state_dict(), "./weights/autoencoder_percept_STL_vae_adv.pkl")

if __name__ == '__main__':
    main()
    torch.randn()