import torch as torch
import torch.autograd as autograd
from torch.autograd import Variable
from random import random, weibullvariate
import opendatasets as od
import os                 # Dient zum lokalen Speichern des Datasets
import numpy as np

import torchvision
from torchvision import datasets
import torch.utils.data as DataLoader
import torchvision.datasets as ImageFolder
import torchvision.transforms as transforms  # Transformieren von Bildern
import matplotlib.pyplot as plt  # plotten von Grafen/ Bildern
from torchvision.utils import make_grid
import torch.nn as nn  # Neuronales Netz
import torch.optim as optim  # Optimierungs-Algorithmen
from torchvision.utils import save_image  # Speichern von Bildern
from matplotlib import image
from tkinter.tix import IMAGE
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

IMAGE_SIZE = 64  # Größe der Bilder
BATCH_SIZE = 64  # Anzahl der Batches
WORKERS = 2  # Anzahl der Kerne beim Arbeiten auf der GPU
# Normalisierung mit 0.5 Mittelwert und Standardabweichung für alle drei Channels der Bilder
NORM = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
WORKERS = 2  # Anzahl der Kerne beim Arbeiten auf der GPU
NUM_EPOCH = 4  # Anzahl der Epochen
LR = 1e-4  # Learningrate
LATENT_SIZE = 100  # Radom Input für den Generator
N_CRITIC = 5
LAMBDA_GP = 10  # Penalty Koeffizient
no_of_channels = 3
cur_step = 0
display_step = 500

data_dir = './data/'
os.makedirs(data_dir, exist_ok=True)  # Anlegen eines Ordners für Bilder

# Erklärung zum Umgang mit Opendata und Kaggle - https://pypi.org/project/opendatasets/
# Datensatz:Anime-Faces werden von Kaggle geladen
# Hierfür wird der User-API-KEY benötigt
# APIKEY {"username":"kimmhl","key":"f585163b4ee30f0a5b44b1a902dc56e6"}
dataset_url = 'https://www.kaggle.com/splcher/animefacedataset'
# Images werden in './animefacedataset' gespeichert
od.download(dataset_url, data_dir)

"""
Ausgabe zu den Ordnern
"""
print(os.listdir(data_dir))  # zeigt Ordner an)

# gibt 10 Bezeichnungen von Bildern aus
print(os.listdir(data_dir+'animefacedataset/images')[:10])


# Transformer
transform = transforms.Compose([
    # Resize der Images auf 64 der kürzesten Seite; Andere Seite wird
    transforms.Resize(IMAGE_SIZE),
    # skaliert, um das Seitenverhältnis des Bildes beizubehalten.
    # Zuschneiden auf die Mitte des Images, sodass ein quadratisches Bild mit 64 x 64 Pixeln entsteht
    transforms.CenterCrop(IMAGE_SIZE),
    # Umwandeln in einen Tensor (Bildern in numerische Werte umwandeln)
    transforms.ToTensor(),
    transforms.Normalize(*NORM)])          # Normalisierung Mean & Standardabweichung von 0.5 für alle Channels
# (Anzahl: 3 für farbige Bilder)
# Pixelwerte liegen damit zwischen (-1;1)

# Dataset
"""
ImageFolder() : Befehl erwartet, dass nach Images nach labeln organisiert sind (root/label/picture.png)
"""
org_dataset = torchvision.datasets.ImageFolder(
    root=data_dir, transform=transform)

# Dataloader
"""
Dataloader():
"""
org_loader = torch.utils.data.DataLoader(org_dataset,              # Dataset (Images)
                                         # Es wird auf Batches trainiert, damit auf Basis eines Batch-Fehlers das NN angepasst wird
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=WORKERS)

# Nutzen der GPU wenn vorhanden, ansonsten CPU


def get_default_device():
    if torch.cuda.is_available():     # Wenn cuda verfügbar dann:
        return torch.device('cuda')   # Nutze Device = Cuda (=GPU)
    else:                         # Ansonsten
        return torch.device('cpu')    # Nutze Device = CPU


# Anzeigen welches Device verfügbar ist
device = get_default_device()
print(device)

# Hilfsklasse zum Verschieben des Dataloaders "org_loader" auf das jeweilige Device


class DeviceDataLoader():

    # Initialisierung
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    # Anzahl der Images pro Batch
    def __len__(self):
        return len(self.dataloader)

    # Erstellt einen Batch an Tensoren nach dem Verschieben auf das Device
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(tensor.to(self.device) for tensor in batch)


# Dataloader auf dem verfügbaren Device
dataloader = DeviceDataLoader(org_loader, device)


def get_noise(n_samples, noise_dim, device=device):
    '''
    Generate noise vectors from the random normal distribution with dimensions (n_samples, noise_dim),
    where
        n_samples: the number of samples to generate based on batch_size
        noise_dim: the dimension of the noise vector
        device: device type can be cuda or cpu
    '''

    return torch.randn(n_samples, noise_dim, 1, 1, device=device)


class Generator(nn.Module):
    def __init__(self, no_of_channels=no_of_channels, noise_dim=LATENT_SIZE, gen_dim=IMAGE_SIZE):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, gen_dim*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gen_dim*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(gen_dim*8, gen_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(gen_dim*4, gen_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gen_dim*2, gen_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gen_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(gen_dim, no_of_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.generator(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, no_of_channels=no_of_channels, disc_dim=IMAGE_SIZE):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(

            nn.Conv2d(no_of_channels, disc_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(disc_dim, disc_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(disc_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(disc_dim * 2, disc_dim * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(disc_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(disc_dim * 4, disc_dim * 8, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(disc_dim * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(disc_dim * 8, 1, 4, 1, 0, bias=False),

        )

    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1).squeeze(1)
        # return output


gen = Generator().to(device)
critic = Discriminator().to(device)

# Gewichtsinitialisierung
#  mean 0 and Standardabweichung 0.02


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, val=0)


gen = gen.apply(weights_init)
critic = critic.apply(weights_init)


# Optimizer

gen_opt = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0, 0.9))
critic_opt = torch.optim.Adam(critic.parameters(), lr=LR, betas=(0, 0.9))

# Gradient Penalty


def gradient_penalty(critic, real_image, fake_image, device=device):
    batch_size, channel, height, width = real_image.shape
    # alpha is selected randomly between 0 and 1
    alpha = torch.rand(batch_size, 1, 1, 1).repeat(
        1, channel, height, width).to(device)
    # interpolated image=randomly weighted average between a real and fake image
    # interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image = (alpha*real_image) + (1-alpha) * fake_image

    # calculate the critic score on the interpolated image
    interpolated_score = critic(interpolatted_image)

    # take the gradient of the score wrt to the interpolated image
    gradient = torch.autograd.grad(inputs=interpolatted_image,
                                   outputs=interpolated_score,
                                   retain_graph=True,
                                   create_graph=True,
                                   grad_outputs=torch.ones_like(
                                       interpolated_score)
                                   )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

# Hilfsfunktionen zur Normalisierng von Tensoren und grafischen Darstellung


def tensor_norm(img_tensors):
    # print (img_tensors)
    # print (img_tensors * NORM [1][0] + NORM [0][0])
    return img_tensors * NORM[1][0] + NORM[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Fake_Images")
    ax.imshow(make_grid(tensor_norm(images.detach()[:nmax]), nrow=8).permute(
        1, 2, 0).cpu())  # detach() : erstellt eine neue "Ansicht",
    # sodass diese Operationen nicht mehr verfolgt werden,orm
    # d. h. der Gradient wird nicht berechnet und der Untergraph
    # wird nicht aufgezeichnet > Speicher wird nicht verwendet


"""
Ordner anlegen für die vom Generator erstellten Images
"""

dir_gen_samples = '../data/outputs/'
#os.makedirs('../outputs/dir_gen_samples', exist_ok=True)
os.makedirs(dir_gen_samples, exist_ok=True)


def saves_gen_samples(idx, random_Tensor):
    # Randomisierter Tensor wird an den Generator übergeben
    fake_img = gen(random_Tensor)
    # Setzen von Bildbezeichnungen für die Fake_Images
    fake_img_name = "gen_img-{0:0=4d}.png".format(idx)
    # Tensor-Normalisierung; Speichern der Fake_Images im Ordner "Outputs/dir_gen_samples/"
    save_image(tensor_norm(fake_img), os.path.join(
        dir_gen_samples, fake_img_name), nrow=8)
    # show_images(fake_img)  # Plotten der Fake_Images
    print("Gespeichert")


def display_images(image_tensor, num_images=25, size=(3, 64, 64)):

    flatten_image = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(flatten_image[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


# Iteration über Epochen
for epoch in range(NUM_EPOCH):

    # Iteration über Batches
    for real_image, _ in tqdm(dataloader):
        cur_batch_size = real_image.shape[0]

        real_image = real_image.to(device)

        # Iteration über Critic (=Discrimiator) Anzahl
        for _ in range(N_CRITIC):

            # Generieren von Radom-Noise
            fake_noise = get_noise(cur_batch_size, LATENT_SIZE, device=device)
            fake = gen(fake_noise)

            # Trainieren des Critics (=Discriminator)
            critic_fake_pred = critic(fake).reshape(-1)
            critic_real_pred = critic(real_image).reshape(-1)

            # Berechnung: gradient penalty auf den realen and fake Images (Generiert durch Generator)
            gp = gradient_penalty(critic, real_image, fake, device)
            critic_loss = -(torch.mean(critic_real_pred) -
                            torch.mean(critic_fake_pred)) + LAMBDA_GP * gp

            # Gradient = 0
            critic.zero_grad()

            # Backprop. + Aufzeichnen dynamischen Graphen
            critic_loss.backward(retain_graph=True)

            # Update Optimizer
            critic_opt.step()

        # Trainieren des Generators: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        gen_loss = -torch.mean(gen_fake)

        # Gradient = 0
        gen.zero_grad()

        # Backprop.
        gen_loss.backward()

        # Update optimizer
        gen_opt.step()

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Step {cur_step}: Generator loss: {gen_loss}, critic loss: {critic_loss}")
            display_images(fake)
            display_images(real_image)
            gen_loss = 0
            critic_loss = 0
            saves_gen_samples(cur_step, fake_noise)
        cur_step += 1
