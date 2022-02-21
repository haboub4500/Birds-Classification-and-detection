import torch

import matplotlib.pyplot as plt

import numpy as np

import itertools

from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    """
    Construction et affichage de la matrice de confusion.
    """
    
    def __init__(self, model, data, train_dataloader, device):
        """
        Constructeur de la classe ConfusionMatrix.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            data: Data.Data
                Les données
            
            train_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données d'entraînement
                
            device: torch.device
                Objet représentant le matériel utilisé
        """
        
        self.model = model
        self.data = data
        self.train_dataloader = train_dataloader
        self.device = device
    
    def prediction(self, model, image):
        """
        Renvoie le nom de l'espèce d'oiseau identifiée par le modèle sur une
        image donnée.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            image: torch.Tensor
                L'image à traiter
                
        SORTIE
            prediction: str
                Le nom de l'espèce identifiée sur l'image
        """
        xb = self.train_dataloader.to_device(image.unsqueeze(0), self.device)
        out = model(xb)
        _, pred = torch.max(out, dim=1)
        prediction = self.data.test_dataset.classes[pred[0].item()]
        
        return prediction
    
    def get_labels_predictions(self):
        """
        Renvoie les étiquettes et les prédictions issues des données de test.
            
        SORTIE
            labels, predictions: tuple
        
            labels: list[str]
                Liste des étiquettes des images des données de test
                
            predictions: list[str]
                Liste des prédictions des images des données de test
        """
        
        predictions = []
        labels = []

        for i in range(len(self.data.test_dataset)):
            images_, labels_ = self.data.test_dataset[i]
            predictions.append(self.prediction(self.model, images_))
            labels.append(self.data.test_dataset.classes[labels_])
             
        return labels, predictions
    
    def plot_confusion_matrix(self, cm, classes, normalize=True, cmap=plt.cm.Blues):
        """
        Etape intermédiaire à l'affichage de la matrice de confusion.
        """
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('true label')
        plt.xlabel('predicted label')
        
    def show_confusion_matrix(self):
        """
        Affiche la matrice de confusion du modèle.
        """
        
        self.model.network.eval()
        
        labels, predictions = self.get_labels_predictions()
        
        cnf_matrix = confusion_matrix(labels, predictions)
        
        np.set_printoptions(precision=2)
        
        plt.figure(figsize=(6, 6))
        self.plot_confusion_matrix(cnf_matrix, self.data.train_dataset.classes)

        plt.show()