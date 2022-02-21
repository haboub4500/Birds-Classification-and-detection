from ImageClassificationBase import ImageClassificationBase

import torchvision.models as models
import torch.nn as nn
import torch

import os
import shutil

class Model(ImageClassificationBase):
    """
    Classe relative au modèle du réseau de neurones utilisé.
    """
    
    def __init__(self, num_classes):
        """
        Constructeur de la classe Model.
        
        ENTREE
            num_classes: int
                Le nombre d'espèces
        """
        
        # Appel du constructeur de la superclasse ImageClassificationBase
        super().__init__()
        
        # Modèle de réseau de neurones
        self.network = self.init_network(num_classes)
    
    def init_network(self, num_classes, name='resnet34'):
        """
        Renvoie l'architecture qui sera utilisée.
        
        ENTREE
            num_classes: int
                Le nombre d'espèces
                
            name: str (optionnel)
                Le nom du modèle Pytorch utilisé
                
        SORTIE
            net: torchvision.models.resnet.ResNet
                Le modèle utilisé
        """
        
        if name == 'resnet18':
            net = models.resnet18(pretrained=True)
        elif name == 'resnet34':
            net = models.resnet34(pretrained=True)
        elif name == 'resnet50':
            net = models.resnet50(pretrained=True)
            
        # Nombre de sorties par défaut
        number_of_features = net.fc.in_features
        
        # On ajoute une couche supplémentaire à ce réseau de neurones pour
        # avoir autant de sorties que d'espèces d'oiseaux
        net.fc = nn.Linear(number_of_features, num_classes)
        
        return net
    
    def forward(self, batch_images):
        """
        Renvoie les valeurs en sortie du réseau de neurones, associées à chaque
        image d'un lot donné.
        
        ENTREE
            batch_images: torch.Tensor
                Un lot d'images
                
        SORTIE
            res: torch.Tensor
                Les valeurs en sortie du réseau de neurones, associées à chaque
                image du lot
        """
        return self.network(batch_images)
    
    def freeze(self):
        """
        Arrête l'apprentissage hors dernière couche.
        """
        
        for param in self.network.parameters():
            param.requires_grad= False
        for param in self.network.fc.parameters():
            param.requires_grad= True
        
    def unfreeze(self):
        """
        Reprend l'apprentissage de toutes les couches.
        """
        
        for param in self.network.parameters():
            param.requires_grad= True
            
    def save_model(self):
        """
        Sauvegarde le modèle.
        """
        
        if not os.path.exists("save"):
            os.mkdir("save")
        else:
            shutil.rmtree("save", ignore_errors=True)
            os.mkdir("save")
                
        torch.save(self.network, 'save/model.pth')
        
    def load_model(self):
        """
        Charge un modèle sauvegardé.
        """
        
        if os.path.isfile('save/model.pth'):
            net = torch.load('save/model.pth')
            net.eval()
            self.network = net
        else:
            print("Erreur : Pas de modèle sauvegardé.")