# Importation des bibliothèques nécessaires
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import cv2
import os
from tempfile import NamedTemporaryFile
from descriptor import glcm, bitdesc, haralick_feat_beta, features_extraction_concat
from distances import retrieve_similar_image

# Fonction pour charger les signatures en fonction du type de descripteur
def load_signatures(descriptor_type):
    if descriptor_type == "GLMC":
        return np.load('glcm_signatures.npy', allow_pickle=True)
    elif descriptor_type == "BIT":
        return np.load('bit_signatures.npy', allow_pickle=True)
    elif descriptor_type == "Haralick":
        return np.load("haralick_signatuees.npy", allow_pickle=True)
    elif descriptor_type == "Combine":
        return np.load("final_combined_signatures.npy", allow_pickle=True)
    else:
        return None

# Fonction principale pour exécuter l'application Streamlit
def main():
    # Configuration de la page
    st.set_page_config(page_title='Feature Extraction', page_icon='🧠')
    st.title('🧠 Feature Extraction')

    # Définir les onglets de l'application
    tab_names = ["Traditionnel", "Avancé"]
    selected_tab = st.sidebar.radio("Select Tab", tab_names, key="selected_tab")

    # Options pour les distances
    distance_options = ["Manhattan", "Euclidean", "Chebyshev", "Canberra"]

    # Si l'onglet sélectionné est "Traditionnel"
    if selected_tab == "Traditionnel":
        st.sidebar.header("Settings for Traditionnel")
        
        # Options de descripteur traditionnel
        traditional_descriptor_options = ["GLMC", "BIT"]
        selected_descriptor = st.sidebar.radio("Descriptor", traditional_descriptor_options, key="trad_descriptor")
        
        # Sélection de la distance
        selected_distance = st.sidebar.radio("Distance", distance_options, key="trad_distance")
        
        # Sélection de la distance maximale
        max_distance = st.sidebar.number_input("Distance maximale", min_value=0.0, value=100.0, key="trad_max_distance")

    # Si l'onglet sélectionné est "Avancé"
    elif selected_tab == "Avancé":
        st.sidebar.header("Settings for Avancé")
        
        # Options de descripteur avancé
        advanced_descriptor_options = ["GLMC", "BIT", "Haralick", "Combine"]
        selected_descriptor = st.sidebar.radio("Descriptor", advanced_descriptor_options, key="adv_descriptor")
        
        # Sélection de la distance
        selected_distance = st.sidebar.radio("Distance", distance_options, key="adv_distance")
        
        # Sélection de la distance maximale
        max_distance = st.sidebar.number_input("Distance maximale", min_value=0.0, value=100.0, key="adv_max_distance")

        # Options de transformation des données
        transform_options = [
            ('No Transform', None),
            ('Rescale', MinMaxScaler()),
            ('Normalization', Normalizer()),
            ('Standardization', StandardScaler())
        ]
        selected_transform_name = st.sidebar.radio("Select Data Transformation", [name for name, _ in transform_options], key="adv_transform")

        # Options de modèles de machine learning
        model_options = [
            ('LDA', LinearDiscriminantAnalysis()),
            ('KNN', KNeighborsClassifier(n_neighbors=10)),
            ('Naive Bayes', GaussianNB()),
            ('Decision Tree', DecisionTreeClassifier()),
            ('SVC', SVC(C=2.5, max_iter=5000, probability=True)),
            ('Random Forest', RandomForestClassifier()),
            ('AdaBoost', AdaBoostClassifier())
        ]
        selected_model_name = st.sidebar.radio("Select Machine Learning Model", [name for name, _ in model_options], key="adv_model")

    # Charger les signatures en fonction du descripteur sélectionné
    signatures = load_signatures(selected_descriptor)
    if signatures is not None:
        total_images = len(signatures)
        st.sidebar.write(f"Nombre total d'images dans la base de données : {total_images}")

        # Afficher l'onglet sélectionné
        st.header(selected_tab)

        # Téléversement de l'image
        st.write("Veuillez téléverser votre image:")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key=f"{selected_tab}_image_upload")

        if uploaded_file is not None:
            # Enregistrer temporairement l'image téléversée
            with NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_image_path = temp_file.name

            # Afficher l'image téléversée
            st.image(uploaded_file, caption='Image téléversée.', use_column_width=True)

            # Afficher les options sélectionnées
            st.write(f"Descripteur sélectionné : {selected_descriptor}")
            st.write(f"Distance sélectionnée : {selected_distance}")
            st.write(f"Distance maximale : {max_distance}")

            if selected_tab == "Avancé":  
                st.write(f"Transformation sélectionnée : {selected_transform_name}")
                st.write(f"Modèle sélectionné : {selected_model_name}")

            # Extraire les caractéristiques de l'image en fonction du descripteur sélectionné
            features = None
            if selected_descriptor == "GLMC":
                features = glcm(temp_image_path)[:6]
            elif selected_descriptor == "BIT":
                features = bitdesc(temp_image_path)[:14]
            elif selected_descriptor == "Haralick":
                features = haralick_feat_beta(temp_image_path)[:14]
            elif selected_descriptor == "Combine":
                features = features_extraction_concat(temp_image_path)

            if features is not None:
                st.write("Descripteur calculé :", [f"{f:.6f}" for f in features])
                st.write(f"Dimension des caractéristiques calculées : {len(features)}")
                st.write("Les caractéristiques calculées :", [f"{f:.6f}" for f in features])

                # Récupérer les images similaires
                sorted_results = retrieve_similar_image(signatures, features, selected_distance.lower(), total_images)

                # Filtrer les résultats en fonction de la distance maximale
                filtered_results = [result for result in sorted_results if result[1] <= max_distance]

                # Sélection du nombre de résultats à afficher
                num_res_options = list(range(1, len(filtered_results) + 1))
                selected_num_res = st.selectbox("Num Res", num_res_options, key=f"{selected_tab}_num_res")

                st.write(f"Nombre total d'images similaires trouvées : {len(filtered_results)}")
                st.write(f"Top {selected_num_res} résultats les plus proches :")

                # Afficher les images similaires
                cols = st.columns(3)
                for i, result in enumerate(filtered_results[:selected_num_res]):
                    col = cols[i % 3]
                    col.write(f"Image : {result[0]}, Distance : {result[1]:.6f}, Label : {result[2]}")
                    similar_image = cv2.imread(result[0])
                    similar_image = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)
                    col.image(similar_image, caption=f"Similar Image (Distance: {result[1]:.6f})", use_column_width=True)

                # Si l'onglet "Avancé" est sélectionné, effectuer une évaluation de modèle
                if selected_tab == "Avancé":
                    # Charger les données pour l'évaluation du modèle
                    data = pd.read_csv('final_signatures_RGB.csv')
                    X = data.iloc[:, :-2]  # Caractéristiques
                    y = data.iloc[:, -2]   # Étiquettes
                    
                    # Encodage des colonnes non numériques
                    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
                    for col in non_numeric_columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])

                    # Remplacer les valeurs infinies par NaN et les remplir avec la moyenne
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X.fillna(X.mean(numeric_only=True), inplace=True)

                    # Encodage des étiquettes
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)

                    # Diviser les données en ensembles d'entraînement et de test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Appliquer la transformation des données sélectionnée
                    selected_transform = next(trans for name, trans in transform_options if name == selected_transform_name)
                    selected_model = next(model for name, model in model_options if name == selected_model_name)

                    if selected_transform is not None:
                        selected_transform.fit(X_train)
                        X_train = selected_transform.transform(X_train)
                        X_test = selected_transform.transform(X_test)

                    # Entraîner le modèle sélectionné
                    selected_model.fit(X_train, y_train)

                    # Prédire les étiquettes sur l'ensemble de test
                    y_pred = selected_model.predict(X_test)

                    # Afficher les métriques d'évaluation du modèle
                    st.write("\n**Model Evaluation Metrics:**")
                    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred) * 100:.2f}%")
                    st.write(f"**F1-Score:** {f1_score(y_test, y_pred, average='macro', zero_division=0) * 100:.2f}%")
                    st.write(f"**Precision:** {precision_score(y_test, y_pred, average='macro', zero_division=0) * 100:.2f}%")
                    st.write(f"**Recall:** {recall_score(y_test, y_pred, average='macro', zero_division=0) * 100:.2f}%")

            # Supprimer le fichier temporaire
            os.remove(temp_image_path)

# Exécuter l'application principale
if __name__ == '__main__':
    main()
