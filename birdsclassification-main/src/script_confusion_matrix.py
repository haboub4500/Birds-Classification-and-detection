"""
Ce script permet d'afficher la matrice de confusion du modèle.

Pour ce faire, il  faut qu'il existe au préalable un modèle sauvegardé de
chemin save/model.pth. S'il n'existe pas, vous pouvez lancer le script
script_training.py qui entraînera puis enregistrera un modèle dans le bon
dossier.
"""

from Main import Main

if __name__ == '__main__':
    
    main = Main()
    
    main.load_model()
    
    main.show_confusion_matrix()