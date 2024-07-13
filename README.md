# bank_scoring
# 


# Objective
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

# Mission

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

## Analyse exploratoire des données

## Prétraitement des datas

-Identification et transformation des variables catégorielles via OneHotEncoder  
-Création de nouvelles variables à partir des variables existantes
-Gestion des classes non équilibrées via la méthode SMOTE
-Traitement des NaN

## Metrics

-Choix des metriques adaptées suivant les modèles entraînés (accuracy, F1 score,...)
-Définition du score métier de type 10*FN + FP (où FN = nombre de FN dans la matrice de confusion pour un seuil donné, FP = nombre de FP) 

## Modélisation 

-Configuration du tunnel ngrok (Mlflow) permettant le tracking lors de l’entraînement des modèles, la visualisation et la comparaison via l’UI de MLFlow, ainsi que le stockage de manière centralisée des modèles

-Entrainement du modèle de référence DummyRegressor
-Entrainement du modèle LightGBM
-Entrainement du modèle XGboost
-Enregistrement des metrics et plots importants
-Choix du meilleur modèle de prédiction: Le choix des valeurs optimales des hyperparamètres est réalisée automatiquement dans le cadre du GridSearchCV

## Features importances
-Features importances globales pour chaque modèle entrainé
-feature importance locale

## Streamlit

L'application Streamlit vise à simplifier le processus d'évaluation du crédit en offrant une interface intuitive et réactive pour prédire et visualiser les scores de crédit des clients à partir d'un modèle pré-entraîné
 
## Déployement de mon application Fastapi sur le cloud via Azure (en se basant sur le dépôt Githup)

## Application Streamlit de simulation d’un scoring client

## Data drift

L'Objectifs de l'Analyse de Data Drift est la détéction de la dérive des données qui reste cruciale pour maintenir la performance d'un modèle de machine learning en production. Si les données utilisées pour entraîner le modèle diffèrent significativement des nouvelles données, le modèle peut devenir moins précis et ses prédictions moins fiables. En identifiant la dérive des données, on peut :

-Prendre des mesures correctives, telles que réentraîner le modèle avec des données plus récentes.
-Identifier des changements dans le comportement des utilisateurs ou dans le processus de génération des données.
-Maintenir la performance et la précision du modèle de manière proactive.
