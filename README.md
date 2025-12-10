# Chatbot Émotionnel – Tkinter  

**Version 1.0.0** 

## À propos de ce projet

Ce projet est une application Python dotée d’une interface graphique Tkinter, conçue pour simuler un chatbot émotionnel intelligent.
Le programme analyse les messages de l’utilisateur, détecte l’émotion dominante (joie, tristesse, colère, peur, amour, etc.) et répond de manière adaptée, en affichant des émoticônes et des couleurs thématiques.

Le chatbot peut fonctionner avec ou sans modèle d’apprentissage automatique.

Si TensorFlow et HuggingFace Datasets sont installés, il utilise un modèle LSTM bidirectionnel pour reconnaître les émotions.
Sinon, il s’appuie sur un système de règles et de réponses prédéfinies stockées dans un fichier JSON.
Ce projet illustre la combinaison entre intelligence artificielle, traitement du langage naturel (NLP) et interface utilisateur interactive.

## Installation

### 1. Cloner le dépôt 

### 2. Installation des dépendances
cd projet
### 3. Vérifier la présence des fichiers nécessaires
Assurez-vous que les fichiers suivants sont dans le même dossier :

chatbot_gui.py
reponse.json
memory.json (optionnel)

### 4. Démarrage du serveur
Installez les bibliothèques Python requises :

pip install tensorflow numpy datasets tkinter

### 5. Lancer l’application
Exécutez le script principal :

python chatbot_gui.py

### 6. Utilisation
Une fenêtre Tkinter s’ouvre :

Saisissez un message dans le champ de texte.
Le chatbot analysera votre message et répondra avec une émotion détectée.
Les couleurs et emojis changent selon l’émotion (joie 😄, tristesse 😢, colère 😡, etc.).

  


