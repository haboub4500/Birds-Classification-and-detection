import torch.nn as nn
import torch.nn.functional as F
import torch

class ImageClassificationBase(nn.Module):
    """
    Calcul de différentes pertes et erreurs au cours de l'entraînement.
    """
    
    def training_step(self, batch):
        """
        Renvoie la valeur de perte pour un lot d'images.
        
        ENTREE
            batch: list
                Liste contenant deux éléments
                
                - images: torch.Tensor
                    Les images que contient le lot
                - labels: torch.Tensor
                    Les étiquettes de chaque image du lot
                    
        SORTIE
            loss: torch.Tensor
                Valeur de perte, c'est-à-dire l'écart entre les prédictions et
                les valeurs réelles (écart que l'on cherche donc à minimiser)
        """
        
        images, labels = batch
        
        out = self(images)
        loss = F.cross_entropy(out, labels)
        
        return loss
    
    def accuracy(self, out, labels):
        """
        Renvoie la proportion de prédictions correctes dans le lot d'images.
        
        ENTREE
            out: torch.Tensor
                Tensor à deux dimensions avec autant de lignes que d'images dans
                un lot, et autant de colonnes que d'espèces d'oiseaux. Plus la
                valeur out[i,j] est grande, plus il est probable que l'image
                d'indice i du lot soit de l'espèce d'indice j.
                
            labels: torch.Tensor
                Les étiquettes de chaque image du lot
                
        SORTIE
            acc: torch.Tensor
                La proportion de prédictions correctes dans le lot d'images
        """
        
        _, preds = torch.max(out, dim=1)
        
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    def validation_step(self, batch):
        """
        Renvoie la valeur de perte et la proportion de prédictions correctes
        associées à un lot d'images.
        
        ENTREE
            batch: list
                Liste contenant deux éléments
                
                - images: torch.Tensor
                    Les images que contient le lot
                - labels: torch.Tensor
                    Les étiquettes de chaque image du lot
        
        SORTIE
            dict: Dict[str, torch.Tensor]
                Un dictionnaire à deux éléments contenant la perte et la précision
                des prédictions, associées au lot d'images
        """
        
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        
        return {"valid_loss": loss.detach(), "valid_acc": acc}
    
    def validation_epoch_end(self, outputs):
        """
        Renvoie la perte moyenne et la précision moyenne obtenues au cours
        de l'epoch.
        
        ENTREE
            outputs: list
                Une liste de dictionnaires renvoyés par la méthode validation_step
                pour chaque lot d'images
                
        SORTIE
            dict: Dict[str, float]
                Un dictionnaire à deux composants contenant la perte moyenne et
                la précision moyenne obtenues au cours de l'epoch
        """
        
        batch_loss = [x["valid_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["valid_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        
        return {"valid_loss": epoch_loss.item(), "valid_acc": epoch_acc.item()}
    
    def epoch_end(self, epoch, epochs, result):
        """
        Affiche divers résultats associés à l'epoch achevé.
        
        ENTREE
            epoch: int
                L'indice de l'epoch achevé
                
            epochs: int
                Le nombre total d'epoch
                
            result: Dict[str, float]
                Des résultats associés à l'epoch achevé
        """
        
        print("Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, valid_loss: {:.4f}, valid_acc: {:.4f}".format(
            epoch+1, epochs, result["lrs"][-1], result["train_loss"],
            result["valid_loss"], result["valid_acc"]))