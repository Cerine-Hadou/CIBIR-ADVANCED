import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Charger les données à partir d'un fichier CSV
file_path = 'final_signatures_RGB.csv'
data = pd.read_csv(file_path)

# Inspecter la forme des données
print("Shape of data:", data.shape)

# Séparer les caractéristiques et les étiquettes
X = data.iloc[:, :-2]  # Toutes les colonnes sauf les deux dernières
y = data.iloc[:, -2]   # La deuxième à la dernière colonne contient les labels

# Identifier les colonnes non numériques
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns:", non_numeric_columns)

# Traiter les colonnes non numériques
for col in non_numeric_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Vérification et nettoyage des données
print("Vérification des valeurs infinies ou aberrantes dans les caractéristiques...")
X = X.replace([np.inf, -np.inf], np.nan)
X.fillna(X.mean(numeric_only=True), inplace=True)

# Vérifier s'il y a encore des valeurs infinies ou aberrantes
if np.any(np.isinf(X)) or np.any(np.isnan(X)):
    print("Il y a encore des valeurs problématiques après le nettoyage.")
else:
    print("Toutes les valeurs sont finies et valides.")

# Assurez-vous que les étiquettes sont sous une forme numérique si elles sont catégorielles
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

transforms = [
    ('NoTransform', None),
    ('Rescale', MinMaxScaler()), 
    ('Normalization', Normalizer()),
    ('Standardization', StandardScaler())
]

# Définir les modèles à évaluer
models = [
    ('LDA', LinearDiscriminantAnalysis()), 
    ('KNN', KNeighborsClassifier(n_neighbors=10)),
    ('Naive Bayes', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVC', SVC(C=2.5, max_iter=5000, probability=True)),
    ('Random Forest', RandomForestClassifier()),
    ('AdaBoost', AdaBoostClassifier())
]

# Métriques d'évaluation
metrics = [
    ('Accuracy', accuracy_score), 
    ('F1-Score', f1_score), 
    ('Precision', precision_score),
    ('Recall', recall_score)
]

# Entraînement et évaluation des modèles
for trans_name, trans in transforms:
    print(f'\nTransformation: {trans_name}\n{"-"*30}')
    
    if trans is not None:
        # Appliquer la transformation
        trans.fit(X_train)
        X_tr = trans.transform(X_train)
        X_te = trans.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test
    
    for metr_name, metric in metrics:
        print(f'\nMétrique: {metr_name}\n{"-"*20}')
        for mod_name, model in models:
            # Entraîner le modèle
            model.fit(X_tr, y_train)
            
            # Prédire les résultats
            y_pred = model.predict(X_te)
            
            # Calculer la métrique
            if metr_name in ['F1-Score', 'Precision', 'Recall']:
                result = metric(y_test, y_pred, average='macro', zero_division=0)
            else:
                result = metric(y_test, y_pred)
                
            print(f'{mod_name}: {result*100:.2f}%')

# Obtenir les noms des classes présentes dans y_test
present_classes = np.unique(y_test)
present_class_names = [label_encoder.classes_[i] for i in present_classes]

# Rapport de classification pour un modèle choisi (par exemple, Random Forest)
chosen_model = RandomForestClassifier()
chosen_model.fit(X_train, y_train)
y_pred_chosen = chosen_model.predict(X_test)
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_chosen, target_names=present_class_names))

# Matrice de confusion pour le modèle choisi
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_chosen))

