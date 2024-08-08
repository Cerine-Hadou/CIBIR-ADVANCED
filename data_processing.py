import cv2
import os
from descriptor import glcm, bitdesc,haralick_feat_beta, features_extraction_concat
import numpy as np

def extract_features(image_path, descriptor_func):
    print(f"Reading image: {image_path}")
    try:
        img = cv2.imread(image_path, 0)
        if img is not None:
            features = descriptor_func(image_path)
            print(f"Extracted features from {image_path}: {features}")
            print(f"Extracted features shape: {len(features)}")
            return features
        else:
            print(f"Failed to read image: {image_path}")
            return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


# # def process_datasets(root_folder):
# #     glcm_features_list = []
# #     bit_features_list = []
# #     haralick_features_list = []
# #     rgb_features_list =  []
    
# #     for root, dirs, files in os.walk(root_folder):
# #         for file in files:
# #             if file.lower().endswith(('.jpg', '.png', '.jpeg')):
# #                 image_rel_path = os.path.join(root, file)
# #                 print(f"Processing file: {image_rel_path}")
# #                 if os.path.isfile(image_rel_path):
# #                     try:
# #                         folder_name = os.path.basename(os.path.dirname(image_rel_path))
#                         #glcm_features = extract_features(image_rel_path, glcm)
#                         #bit_features = extract_features(image_rel_path, bitdesc)
#                         # haralick_features =haralick_feat_beta(image_rel_path)
#                         # RGB_features=  features_extraction_concat(image_rel_path)
#                         #if glcm_features is not None:
#                             #glcm_features = glcm_features + [folder_name, image_rel_path]
#                             #glcm_features_list.append(glcm_features)
#                         #if bit_features is not None:
#                             #bit_features = bit_features + [folder_name, image_rel_path]
#                             #bit_features_list.append(bit_features)
#                         # if haralick_features is not None :
#                         #     haralick_features = haralick_features + [folder_name, image_rel_path]
#                         #     haralick_features_list.append(haralick_features)
#                 #         RGB_features =features_extraction_concat(image_rel_path)

#                 #         if RGB_features is not None:
#                 #             RGB_features = RGB_features + [folder_name, image_rel_path]
#                 #             rgb_features_list.append(RGB_features)

#                 #     except Exception as e:
#                 #         print(f"Error processing file {image_rel_path}: {e}")
#                 # else:
#                 #     print(f"File does not exist: {image_rel_path}")
    
#     #glcm_signatures = np.array(glcm_features_list)
#     #bit_signatures = np.array(bit_features_list)
#     # haralick_signatures = np.array(haralick_features_list)
    
#     #print(f"GLCM features shape: {glcm_signatures.shape}")
#     #print(f"BIT features shape: {bit_signatures.shape}")
#     # print(f"Haralick features shape: {haralick_features_list}")
    
#     #np.save('glcm_signatures.npy', glcm_signatures)
#     #np.save('bit_signatures.npy', bit_signatures)
#     # np.save('haralick_signatures.npy', haralick_signatures)
#     # print('Successfully stored!')
#     # rgb_signatures=np.array(rgb_features_list)
#     #rgb_signatures1=np.array(rgb_features_list)
#     #np.save('rgb_signatures1.npy', rgb_signatures1 )
#     # np.save('rgb_signatures.npy', rgb_signatures)
#     # print('Successfully stored!')


# def process_datasets(root_folder):
#     features_RGBL = []  # Liste pour stocker les caractéristiques RGB
#     # Parcourir les fichiers dans le dossier racine
#     for root, dirs, files in os.walk(root_folder):
#         for file in files:
#             if file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Vérifier les extensions de fichiers d'image
#                 image_rel_path = os.path.join(root, file)  # Construire le chemin relatif de l'image
#                 print(f"Processing file: {image_rel_path}")  # Afficher le fichier en cours de traitement
#                 if os.path.isfile(image_rel_path):  # Vérifier si le fichier existe
#                     try:
#                         folder_name = os.path.basename(root)  # Obtenir le nom du dossier actuel
#                         features_RGB = features_extraction_concat(image_rel_path)
#                         if features_RGB is not None:
#                             features_RGB += [folder_name, image_rel_path]  # Ajouter des informations supplémentaires
#                             features_RGBL.append(features_RGB)  # Ajouter les caractéristiques RGB à la liste
#                             print(f"Extracted features from {image_rel_path}: {features_RGB}")  # Afficher les caractéristiques extraites
#                     except Exception as e:
#                         print(f"Error processing file {image_rel_path}: {e}")  # Afficher l'erreur en cas d'exception
#                 else:
#                     print(f"File does not exist: {image_rel_path}")  # Message d'erreur si le fichier n'existe pas
 
   
#     if features_RGBL:
#         signatures = np.array(features_RGBL)  
#         filename = f'rgb_signatures_{os.path.basename(root_folder)}.npy'
#         np.save(filename, signatures) 
#         print(f'Successfully stored {filename}!')  
#     else:
#         print("No features extracted to store.")
 
# def signatures_stock():
#     datasets = [
       
#         'Projet1_Dataset/Projet1_Dataset/Iris',
#         'Projet1_Dataset/Projet1_Dataset/KTH-TIPS2-a',
      
       
        
       
#         # 'Projet1_Dataset/Projet1_Dataset/Outex24',
#         # 'Projet1_Dataset/Projet1_Dataset/COVID-CT-master'

#         # 'Projet1_Dataset/Projet1_Dataset/Glaucoma/REFUGE/Glaucoma',2
#         # 'Projet1_Dataset/Projet1_Dataset/Satelite_dataset',2
       
#     ]
 
#     for dataset in datasets:
#         print(f"Processing dataset: {dataset}")
#         process_datasets(dataset)  
#     reunite_signatures(datasets)
 
# def reunite_signatures(datasets):
#     all_signatures = []
#     for dataset in datasets:
#         filename = f'rgb_signatures_{os.path.basename(dataset)}.npy'
#         if os.path.exists(filename):
#             signatures = np.load(filename, allow_pickle=True)
#             all_signatures.append(signatures)
#         else:
#             print(f"Signature file not found for dataset: {dataset}")
 
#     if all_signatures:
#         combined_signatures = np.concatenate(all_signatures, axis=0)
#         np.save('combined_rgb_signatures.npy', combined_signatures)  
#         print('Successfully stored all combined RGB signatures in combined_rgb_signatures.npy.')
#     else:
#         print("No signatures to combine.")
 

# signatures_stock()

def process_datasets(root_folder):
    features_RGBL = []  
  
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg','.bmp')): 
                image_rel_path = os.path.join(root, file)
                print(f"Processing file: {image_rel_path}")  
                if os.path.isfile(image_rel_path): 
                    try:
                        folder_name = os.path.basename(root) 
                        features_RGB = features_extraction_concat(image_rel_path)
                        if features_RGB is not None:
                            features_RGB += [folder_name, image_rel_path] 
                            features_RGBL.append(features_RGB)  
                            print(f"Extracted features from {image_rel_path}: {features_RGB}")  
                    except Exception as e:
                        print(f"Error processing file {image_rel_path}: {e}")  
                else:
                    print(f"File does not exist: {image_rel_path}") 
    if features_RGBL:
        signatures = np.array(features_RGBL)  
        filename = f'rgb_signatures_{os.path.basename(root_folder)}.npy'
        np.save(filename, signatures) 
        print(f'Successfully stored {filename}!')  
    else:
        print("No features extracted to store.")

def signatures_stock():
    datasets_group1 = [
        'Projet1_Dataset/Projet1_Dataset/Wildfire_detection',
        'Projet1_Dataset/Projet1_Dataset/Iris',
        'Projet1_Dataset/Projet1_Dataset/KTH-TIPS2-a',
        
    ]
    
    datasets_group2 = [
        'Projet1_Dataset/Projet1_Dataset/Glaucoma/REFUGE/Glaucoma',
        'Projet1_Dataset/Projet1_Dataset/Satelite_dataset',
        
    ]
    
    datasets_group3 = [
        'Projet1_Dataset/Projet1_Dataset/Outex24',
        'Projet1_Dataset/Projet1_Dataset/COVID-CT-master',
    ]

    print("Processing Group 1 datasets...")
    for dataset in datasets_group1:
        print(f"Processing dataset: {dataset}")
        process_datasets(dataset)
    
    print("Processing Group 2 datasets...")
    for dataset in datasets_group2:
        print(f"Processing dataset: {dataset}")
        process_datasets(dataset)

    print("Processing Group 3 datasets...")
    for dataset in datasets_group3:
        print(f"Processing dataset: {dataset}")
        process_datasets(dataset)

    print("Combining signatures for Group 1...")
    reunite_signatures(datasets_group1, 'combined_rgb_signatures1.npy')
    
    print("Combining signatures for Group 2...")
    reunite_signatures(datasets_group2, 'combined_rgb_signatures2.npy')
    
    print("Combining signatures for Group 3...")
    reunite_signatures(datasets_group3, 'combined_rgb_signatures3.npy')
    

    print("Combining all combined signature files into one...")
    combine_signature_files(['combined_rgb_signatures1.npy', 'combined_rgb_signatures2.npy', 'combined_rgb_signatures3.npy'], 'final_combined_signatures.npy')

def reunite_signatures(datasets, output_filename):
    all_signatures = []
    for dataset in datasets:
        filename = f'rgb_signatures_{os.path.basename(dataset)}.npy'
        if os.path.exists(filename):
            signatures = np.load(filename, allow_pickle=True)
            all_signatures.append(signatures)
        else:
            print(f"Signature file not found for dataset: {dataset}")
 
    if all_signatures:
        combined_signatures = np.concatenate(all_signatures, axis=0)
        np.save(output_filename, combined_signatures)  
        print(f'Successfully stored all combined RGB signatures in {output_filename}.')
    else:
        print("No signatures to combine.")

def combine_signature_files(file_list, output_filename):
    all_signatures = []
    for file in file_list:
        if os.path.exists(file):
            try:
                signatures = np.load(file, allow_pickle=True)
                all_signatures.append(signatures)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File {file} does not exist.")
    
    if all_signatures:
        final_combined_signatures = np.concatenate(all_signatures, axis=0)
        np.save(output_filename, final_combined_signatures)
        print(f'All signature files have been successfully combined into {output_filename}.')
    else:
        print("No signature files to combine.")

signatures_stock()