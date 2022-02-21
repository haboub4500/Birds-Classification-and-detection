import torch

class Training:
    
    def evaluate(self, model, valid_dataloader):
        """
        Renvoie la perte moyenne et la précision moyenne obtenues lors de la
        phase de validation du modèle.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            valid_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données de validation
                
        SORTIE
            dict: Dict[str, float]
                Un dictionnaire à deux composants contenant la perte moyenne
                et la précision moyenne obtenues au cours de la validation
        """
        
        model.eval() # Les coefficients du réseau ne sont plus mis à jour
        outputs = [model.validation_step(batch) for batch in valid_dataloader]
        
        return model.validation_epoch_end(outputs)

    def get_lr(self, optimizer):
        """
        Renvoie le taux d'apprentissage lors de l'entraînement.
        
        ENTREE
            optimizer: torch.optim.adamax.Adamax
                La fonction d'optimisation
                
        SORTIE
            lr: float
                Le taux d'apprentissage
        """
        
        for param_group in optimizer.param_groups:
            return param_group["lr"]
    
    def fit_one_cycle(self, epochs, max_lr, model, train_dataloader,
                      valid_dataloader, weight_decay=0,
                      opt_func=torch.optim.Adam):
        """
        Cette méthode permet d'entraîner le réseau de neurones sur un nombre
        d'epoch donné. Voilà comment se déroule chaque epoch :
            - On parcourt les données d'entraînement pour mettre à jour les
            coefficients du réseau de neurones
            - On fige ces coefficients
            - On parcourt les données de validation pour mesurer les performances
            du réseau de neurones
            - On stocke ces résultats pour suivre l'évolution de notre réseau
        
        ENTREE
            epochs: int
                Le nombre total d'epoch
                
            max_lr: float
                Le taux d'apprentissage maximum
                
            model: Model.Model
                Le modèle de réseau de neurones
                
            train_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données d'entraînement
                
            valid_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données de validation
                
            weight_decay: float (optionnel)
                Coefficient modifiant le résultat de la parte
                
            opt_func: torch.optim.adamax.Adamax (optionnel)
                La fonction d'optimisation
                
        SORTIE
            history: list
                Une liste des résultats obtenus à la fin de chaque epoch  
        """
        
        torch.cuda.empty_cache()
    
        history = []
    
        # Fonction d'optimisation
        opt = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        
        # Gestion du taux d'apprentissage
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=epochs,
                                               steps_per_epoch=len(train_dataloader))
    
        for epoch in range(epochs):
            
            # On autorise la modification des coefficients du réseau
            model.train()
            
            train_loss = [] # Liste des pertes pour chaque lot
            lrs = [] # Liste des LR pour chaque lot
            
            for batch in train_dataloader:
                loss = model.training_step(batch)
                train_loss.append(loss)
                loss.backward()
                
                opt.step()
                opt.zero_grad()
            
                lrs.append(self.get_lr(opt))
                sched.step()
            
            # On évalue le réseau (en arrêtant l'apprentissage, cf méthode
            # evaluate)
            result = self.evaluate(model, valid_dataloader)
            
            # Au dictionnaire result, on ajoute deux nouvelles valeurs, la
            # perte moyenne et le taux d'apprentissage moyen
            result["train_loss"] = torch.stack(train_loss).mean().item()
            result["lrs"] = lrs
            
            # On affiche les résultats de cette epoch
            model.epoch_end(epoch, epochs, result)
            
            # On stocke ces résultats
            history.append(result)
            
        return history
    
    def train_last_layer(self, model, train_dataloader, valid_dataloader,
                         history, epochs=1, max_lr=5*10e-5,
                         weight_decay = 10e-4, opt_func = torch.optim.Adamax):
        """
        Entraîne la dernière couche du réseau de neurones.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            train_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données d'entraînement
                
            valid_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données de validation
                
            history: list
                Les résultats des différentes epoch
                
            epochs: int (optionnel)
                Le nombre total d'epoch
                
            max_lr: float (optionnel)
                Le taux d'apprentissage maximum
                
            weight_decay: float (optionnel)
                Coefficient modifiant le résultat de la parte
                
            opt_func: torch.optim.adamax.Adamax (optionnel)
                La fonction d'optimisation
        """
        
        model.freeze()
        history += self.fit_one_cycle(epochs, max_lr, model,
                                               train_dataloader,
                                               valid_dataloader,
                                               weight_decay=weight_decay, 
                                               opt_func=opt_func)
        
    def train_all_layers(self, model, train_dataloader, valid_dataloader,
                         history, epochs=1, max_lr=10e-5,
                         weight_decay = 10e-4, opt_func = torch.optim.Adamax):
        """
        Entraîne toutes les couches du réseau de neurones.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            train_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données d'entraînement
                
            valid_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données de validation
                
            history: list
                Les résultats des différentes epoch
                
            epochs: int (optionnel)
                Le nombre total d'epoch
                
            max_lr: float (optionnel)
                Le taux d'apprentissage maximum
                
            weight_decay: float (optionnel)
                Coefficient modifiant le résultat de la parte
                
            opt_func: torch.optim.adamax.Adamax (optionnel)
                La fonction d'optimisation
        """
        
        model.unfreeze()
        history += self.fit_one_cycle(epochs, max_lr, model,
                                               train_dataloader,
                                               valid_dataloader,
                                               weight_decay=weight_decay, 
                                               opt_func=opt_func)
        
    def show_max_acc(self, history):
        """
        Affiche la précision maximale obtenue lors de l'entraînement.
    
        ENTREE
            history: list
                Les résultats des différentes epoch
        """
        
        print("During our training, the maximum accuracy obtained by our model is : " ,
        round(max([history[i]["valid_acc"] for i in range(len(history))]) , 3))
        
    def train_model(self, model, train_dataloader, valid_dataloader):
        """
        Entraîne le réseau de neurones.
        
        ENTREE
            model: Model.Model
                Le modèle de réseau de neurones
                
            train_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données d'entraînement
                
            valid_dataloader: torch.utils.data.dataloader.Dataloader
                Itérateur Pytorch sur les données de validation
        """
        
        history = []
        
        self.train_last_layer(model, train_dataloader, valid_dataloader, history)
        self.train_all_layers(model, train_dataloader, valid_dataloader, history)
        
        self.show_max_acc(history)