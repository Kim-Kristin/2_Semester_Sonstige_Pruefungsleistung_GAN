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
import opendatasets as od
import os                 # Dient zum lokalen Speichern des Datasets
import numpy as np
import torch as torch
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
