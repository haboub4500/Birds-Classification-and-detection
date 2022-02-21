import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torchvision.utils import make_grid

class Data():
    """
    Chargement et affichage des données.
    """
    
    # Chemins vers la base de données
    train_path = r"database\train"
    valid_path = r"database\valid"
    test_path = r"database\test"
    
    transform_dataset = T.Compose([
        T.Resize((128,128)),
        T.RandomHorizontalFlip(),
        T.ToTensor()])
    
    # Chargement des données
    train_dataset = torchvision.datasets.ImageFolder(root=train_path,
                                                     transform=transform_dataset)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_path,
                                                     transform=transform_dataset)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path,
                                                    transform=transform_dataset)
    
    number_of_species = len(train_dataset.classes)
    
    batch_size = 32

    # Mise en forme des données
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=False)
    
    def show_img(self):
        """
        Affiche quelques images de la base de données.
        """
        
        for images, labels in self.train_dataloader:
            # images = tensor de dimensions (32, 3, 128, 128) car on a des
            # batch de 32 images, et on stocke, pour chaque image, les trois
            # composantes RGB, ce qui donne 3 tensor (128,128) par image
            
            # labels = tensor à 32 éléments
            # Chaque élément est un entier compris entre 0 et 4
            
            fig, ax = plt.subplots(figsize=(10,10))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(images[:32], nrow=8).permute(1,2,0))
            
            break
