"""
Ce script permet d'entraîner un modèle puis de le sauvegarder.

Le modèle est sauvegardé dans le dossier save, sous le nom de model.pth. Si 
le dossier save n'existe pas, alors il est créé. Si ce dossier existe et qu'un
fichier model.pth existe déjà, alors il est remplacé par le nouveau modèle
entraîné.
"""

from Main import Main

if __name__ == '__main__':
    
    main = Main()
    
    main.launch_training()
    
    main.save_model()