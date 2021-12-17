"""
Wasserstein GAN
Vorteile:
    - bessere Traings-Stabilität als DCGAN
    -
Nachteile:
    - Längeres training

 Idee beim GAN:
    - Distanz zwischen prob generate und prob realistic soll so klein
    wie möglich sein, um möglichst realistische Bilder zu generieren

Problem:
    - Wie wird die Distanz zwischen den beiden prob Distributionen
    beschrieben/ definiert

Lösung:
    - Wasserstein Distance -> WGAN nutzt Wasserstein Distance für den Loss
    (- Kullback-Leiber (KL) divergence)
    (- Jensen-Shannon (JS) divergence) <- Equivalent zum GAN Loss
        -> Hat Gradienten Probleme die das Training instabil werden lassen

WGAN:
    - Disc. Maximierung von Expression
    - Gen.  Minimierung von Expression

Default- Werte im WGAN
    - LR = 0.00005
    - Clipping Parameter = 0.01
    - Batch Size = 64
    - n_critic = 5 (Anzahl der Iterationen von critic pro generation iteration)

"""


# Pakete importieren
# Import Pakete
# Dient zum Laden des Dataset aus opensource-Quellen (z.B. Kaggle)
from random import weibullvariate
import opendatasets as od
import os                 # Dient zum lokalen Speichern des Datasets
import numpy as np
import torch as t
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
import torch.nn.functional as F  # Loss

# Definieren von Parametern

IMAGE_SIZE = 64  # Größe der Bilder
BATCH_SIZE = 64  # Anzahl der Batches
WORKERS = 2  # Anzahl der Kerne beim Arbeiten auf der GPU
# Normalisierung mit 0.5 Mittelwert und Standardabweichung für alle drei Channels der Bilder
NORM = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
NUM_EPOCH = 20  # Anzahl der Epochen
LR = 0.00005  # Learningrate
LATENT_SIZE = 100  # Radom Input für den Generator
N_CRTIC = 5
WEIGHT_CLIPPING = 0.01


# https://jovian.ai/ahmadyahya11/pytorch-gans-anime
# Ordner für den Download anlegen
data_dir = '../data/'
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
print(os.listdir(data_dir))  # zeigt Ordner an

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
Dataloader(): ermöglicht zufällige Stichproben der Daten auszugeben;
Dient dazu, dass das Modell nicht mit dem gesamten Dataset umgehen muss > Training effizienter
"""
org_loader = t.utils.data.DataLoader(org_dataset,              # Dataset (Images)
                                     # Es wird auf Batches trainiert, damit auf Basis eines Batch-Fehlers das NN angepasst wird
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=WORKERS)

# Nutzen der GPU wenn vorhanden, ansonsten CPU


def get_default_device():
    if t.cuda.is_available():     # Wenn cuda verfügbar dann:
        return t.device('cuda')   # Nutze Device = Cuda (=GPU)
    else:                         # Ansonsten
        return t.device('cpu')    # Nutze Device = CPU


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
org_loader = DeviceDataLoader(org_loader, device)


class Generator(t.nn.Module):
    """
    Generator 1 Input Layer; 3 Hidden Layer ; 1 Output Layer
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Output = (inputsize - 1)*stride - 2*padding + (kernelsite-1)+1
            # ConvTranspose2d hilft dabei aus einem kleinen
            nn.ConvTranspose2d(LATENT_SIZE, IMAGE_SIZE*8, 4, 1, 0, bias=False),
            # Tensor einen größeren Tensor zu erstellen (Bezogen auf Channels)
            nn.BatchNorm2d(IMAGE_SIZE*8),
            nn.ReLU(inplace=True),  # Relu lässt keine negativen werte zu

            nn.ConvTranspose2d(IMAGE_SIZE*8, IMAGE_SIZE * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(IMAGE_SIZE*4, IMAGE_SIZE * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(IMAGE_SIZE*2, IMAGE_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(IMAGE_SIZE, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # (-1 und 1) ; Tanh wird häufig verwendet da eine begrenzte Aktivierung es dem Modell ermöglicht,
            # schneller zu lernen. (https://arxiv.org/pdf/1511.06434.pdf S. 3)

            # Output: 3 x 64 x 64
        )

    # Feedforward
    def forward(self, input):
        output = self.generator(input)
        return output


# Erstellen des Generators
NN_Generator = Generator().to(device)
print(NN_Generator)


class Discriminator (t.nn.Module):
    """
    Diskriminator 1 Input Layer; 3 Hidden Layer ; 1 Output Layer
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # Output = ((inputsize) + 2*padding + (kernelsite-1)-1/stride) -1
            # conv2d hilft dabei aus einem großem Tensor einen kleinen Tensor zu stellen
            nn.Conv2d(3, IMAGE_SIZE, 4, 2, 1, bias=False),
            # Leaky RELU lässt negative Werte zu (nicht wie RELU); Neuronen werden somit nicht auf Null gesetzt
            nn.LeakyReLU(0.2, inplace=True),
            # Hilft dem Generator, da dieser nur "Lernen" kann wenn er vom Diskriminator einen Gradienten erhält

            # state size. (IMAGE_SIZE) x 32 x 32
            nn.Conv2d(IMAGE_SIZE, IMAGE_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (IMAGE_SIZE*2) x 16 x 16
            nn.Conv2d(IMAGE_SIZE * 2, IMAGE_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.1),  #Dropout

            # state size. (IMAGE_SIZE*4) x 8 x 8
            nn.Conv2d(IMAGE_SIZE * 4, IMAGE_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(IMAGE_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),

            # state size. (IMAGE_SIZE*8) x 4 x 4
            nn.Conv2d(IMAGE_SIZE * 8, 1, 4, 1, 0, bias=False),

            # Sigmoid Aktivierungsfunktion
            nn.Sigmoid())  # (Werte zwischen 0 und 1); Sigmoid wird verwendet, um zu erkenen wie weit die generierten Bilder von den orginalen abweichen

    def forward(self, input):
        output = self.discriminator(input)
        output = output.view(output.size(0), -1)
        return output


# Erstellen des Diskriminators
NN_Discriminator = Discriminator.to(device)
print(NN_Discriminator)


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
    # sodass diese Operationen nicht mehr verfolgt werden,
    # d. h. der Gradient wird nicht berechnet und der Untergraph
    # wird nicht aufgezeichnet > Speicher wird nicht verwendet


# Radom Tensor

# Generator --> Input: Random Tensor
# Generator --> Output: Fake-Images (Batchsize, data_dim, Pixel, Pixel)
random_Tensor = t.randn(BATCH_SIZE, LATENT_SIZE, 1, 1, device=device)
fake_images = NN_Generator(random_Tensor)
print(fake_images.shape)

show_images(fake_images)


# Trainieren des Generators

def gen_train(Gen_Opt):

    #Gradienten = 0
    Gen_Opt.zero_grad()

    # Generierung von Fake-Images
    fake_img = NN_Generator(random_Tensor)

    # Übergeben der Fakes-Images an den Diskriminator (Versuch den Diskriminator zu täuschen)
    pred = NN_Discriminator(fake_img)
    # Torch.ones gibt einen tensor zurück welcher nur den Wert 1 enthält, und dem Shape Size = BATCH_SIZE
    #target = t.ones(BATCH_SIZE, 1, device=device)
    # loss = F.binary_cross_entropy(pred, target)  # loss_func(pred, target)

    loss = -t.mean(pred)

    loss.backward()
    Gen_Opt.step()

    # Backprop./ Update der Gewichte des Generators
    # loss.backward()
    # Gen_Opt.step()

    #print("Training Gen")
    return loss.item()


# Trainieren des Diskriminators

def disc_train(real_images, Dis_Opt):

    #Gradienten = 0
    Dis_Opt.zero_grad()

    # ????
    #data = real_images.to(device)
    #cur_batch_size = data.shape[0]
    #noise = t.randn(cur_batch_size, 128, 1, 1).to(device)

    fake = NN_Generator(random_Tensor).to(device)

    # Reale Bilder werden an den Diskriminator übergeben
    pred_real = NN_Discriminator(real_images)
    # print(pred_real.size())

    # Kennzeichnen der realen Bilder mit 1
    #target_real = t.ones(real_images.size(0), 1, device=device)
    # print(target_real.size())

    # Berechnung des Losses mit realen Bildern
    #loss_real = F.binary_cross_entropy(pred_real, target_real)
    #real_score = t.mean(pred_real).item()

    """
    2 Erstellen von Fake_Bildern
    """
    # Generierung von Fakeimages
    #fake_img = NN_Generator(random_Tensor).to(device)

    """
    3 Trainieren des Diskriminators auf den erstellten Fake_Bildern
    """
    # Fake Bilder werden an den Diskriminator übergeben
    pred_fake = NN_Discriminator(fake).to(device)

    # Kennzeichnen der Fake-Bilder mit 0
    #target_fake = t.zeros(fake_img.size(0), 1, device=device)

    # Loss Function - Fehler des Fake-Batch wird berechnet
    #loss_fake = F.binary_cross_entropy(pred_fake, target_fake)
    #fake_score = t.mean(pred_fake).item()

    # Berechnung des Gesamt-Loss von realen und fake Images
    #loss_sum = loss_real + loss_fake

    loss_critic = -(t.mean(pred_real)-t.mean(pred_fake))

    loss_critic.backward(retain_graph=True)

    Dis_Opt.step()

    #print("Training disc")
    return loss_critic.item()  # real_score, fake_score


"""
Ordner anlegen für die vom Generator erstellten Images
"""

dir_gen_samples = '../outputs/dir_gen_samples'
os.makedirs('../outputs/dir_gen_samples', exist_ok=True)
# os.makedirs(dir_gen_samples,exist_ok=True)

"""
Funktion zum Speichern der generierten Bilder
"""


def saves_gen_samples(idx, random_Tensor):
    # Randomisierter Tensor wird an den Generator übergeben
    fake_img = NN_Generator(random_Tensor)
    # Setzen von Bildbezeichnungen für die Fake_Images
    fake_img_name = "gen_img-{0:0=4d}.png".format(idx)
    # Tensor-Normalisierung; Speichern der Fake_Images im Ordner "Outputs/dir_gen_samples/"
    save_image(tensor_norm(fake_img), os.path.join(
        dir_gen_samples, fake_img_name), nrow=8)
    show_images(fake_img)  # Plotten der Fake_Images
    print("Gespeichert")


# Aufruf der Funktion
saves_gen_samples(0, random_Tensor)

"""
Zentrale Trainings-Funktion
"""


def train(NN_Discriminator, NN_Generator, NUM_EPOCH, LR, start_idx=1):
    t.cuda.empty_cache()  # leert den Cache, wenn auf der GPU gearbeitet wird

    NN_Discriminator.train()
    NN_Generator.train()

    # Listen für Übersicht des Fortschritts
    R_Score = []
    F_Score = []
    G_losses = []
    D_losses = []

    Gen_Opt = t.optim.RMSprop(NN_Generator.parameters(),
                              lr=LR)
    Dis_Opt = t.optim.RMSprop(NN_Discriminator.parameters(),
                              lr=LR)

    # Iteration über die Epochen
    for epoch in range(0, NUM_EPOCH):

        # Iteration über die Bilder
        for i, (img_real, _) in enumerate(org_loader):

            for _ in range(N_CRTIC):
                # Trainieren des Diskrimniators
                d_loss, real_score, fake_score = disc_train(img_real, Dis_Opt)

                for p in NN_Discriminator.parameters():
                    p.data.clamp_(-WEIGHT_CLIPPING, WEIGHT_CLIPPING)

            # Trainieren des Generators
            g_loss = gen_train(Gen_Opt)

            Count = i  # Index/ Iterationen zählen
            print("index:", i, "D_loss:", d_loss, "G_Loss:", g_loss)

        # Speichern des Gesamtlosses von D und G und der Real und Fake Scores
        D_losses.append(d_loss)
        G_losses.append(g_loss)
        R_Score.append(real_score)
        F_Score.append(fake_score)

        # Ausgabe EPOCH, Loss: G und D, Scores: Real und Fake
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, NUM_EPOCH, g_loss, d_loss, real_score, fake_score))

        # Speichern der generierten Samples/ Images
        saves_gen_samples(epoch+start_idx, random_Tensor)

    return G_losses, D_losses, R_Score, F_Score
