import torch

class DeviceDataLoader():
    
    def __init__(self, dataloader):
        """
        Constructeur de la classe DeviceDataLoader.
        
        ENTREE
            dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur des données
        """
        
        # Itérateur Pytorch sur des données
        self.dataloader = dataloader
        
        # Matériel utilisé
        self.device = self.get_device()
        
    def __iter__(self):
        """
        Fonction d'itération.
        """
        
        for x in self.dataloader:
            yield self.to_device(x, self.device)
            
    def __len__(self):
        """
        Renvoie le nombre de données.
        
        SORTIE
            n: int
                Le nombre de données
        """
        
        return len(self.dataloader)
    
    def to_device(self, data, device):
        """
        Transfère les données dans le matériel.
        """
        
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    
    def get_device(self):
        """
        Renvoie le matériel utilisé.
        
        SORTIE
            device: torch.device
                Objet représentant le matériel utilisé
        """
        
        if torch.cuda.is_available():
            return torch.device("cuda") # Avec architecture CUDA
        else:
            return torch.device("cpu") # Sans architecture CUDA