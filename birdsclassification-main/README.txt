Projet 12 : Solution numérique pour le suivi du dérangement de l'avifaune en
rade de Lorient

Auteurs :
    JUGE Sven
    OUEDRAOGO Souleymane
    SEZGIN Selman

Ce README est un guide d'utilisation du code relatif à la partie
classification du projet.

1. Que contient le dossier src ?
    - database : le dossier contenant la base de données pour l'entraînement
    - img : le dossier contenant les images que l'on veut prédire
    - save : le dossier contenant le modèle sauvegardé
    - Les classes Main, Data, DeviceDataLoader, Model, ImageClassificationBase,
    Train et ConfusionMatrix 
    - Les scripts Python script_training, script_prediction et
    script_confusion_matrix
    
Pour comprendre le rôle de ces classes et de ces scripts, vous pouvez consulter
le code source et sa documentation.

2. Comment entraîner et sauvegarder un réseau de neurones ?
    - Lancer le script src/script_training.py
    - Attendre la fin de l'entraînement
    - Un fichier contenant le modèle entraîné et nommé "model.pth" sera
    sauvegardé dans le dossier "src/save"
    
Remarques :
    - Si le dossier "src/save" n'existe pas, alors il est créé
    - Si le dossier "src/save" existe et contient déjà un fichier "model.pth",
    alors ce fichier est supprimée et remplacée par le nouveau modèle
    
3. Comment effectuer la prédiction d'images ?
    - Créer un dossier "img" dans "src" et y insérer les images voulues
    - Lancer le script src/script_prediction.py
    - Un fichier predictions.csv sera créé avec la prédiction pour chaque
    image
    
Remarques :
	- S'assurer qu'il existe un modèle sauvegardé (src/save/model.pth)
	- Si un fichier predictions.csv existe déjà, il est préalablement supprimé
	avant d'être remplacé par le nouveau
	- Respecter la nomenclature des images : pic_id1_id2.jpeg où
    	id1 est un identifiant à trois chiffres, égal à l'identifiant de
    	l'image sur laquelle se trouve l'oiseau
    	
    	id2 est un identifiant à deux chiffres, égal à l'identifiant de
    	l'oiseau sur l'image