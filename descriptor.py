from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo
import cv2
import numpy as np

def glcm(image_path):
    data = cv2.imread(image_path, 0)
    co_matrix = graycomatrix(data, [1], [np.pi/4], None, symmetric=False, normed=False)
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    features = [np.float32(dissimilarity), np.float32(cont), np.float32(corr), np.float32(ener), np.float32(asm), np.float32(homo)]
    print(f"GLCM features shape: {len(features)}")
    return features
def glcm_beta(data):
    co_matrix = graycomatrix(data, [1], [np.pi/4], None,symmetric=False, normed=False )
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, cont, corr, ener, asm, homo]

def bitdesc(image_path):
    data = cv2.imread(image_path, 0)
    features = bio_taxo(data)
    features = [np.float32(feature) for feature in features]
    required_length = 14  
    if len(features) < required_length:
        features += [np.float32(0)] * (required_length - len(features))
    print(f"BIT features shape (after padding): {len(features)}")
    return features

def bitdesc_beta(data):
    features = bio_taxo(data)
    features = [np.float32(feature) for feature in features]
    required_length = 14  
    if len(features) < required_length:
        features += [np.float32(0)] * (required_length - len(features))
    print(f"BIT features shape (after padding): {len(features)}")
    return features


# def haralick_feat(image_path):
#     data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     haralick_features = haralick_feat(data)
#     features_mean = haralick_features.mean(axis=0)
#     additional_feature = np.var(features_mean)
#     features = np.append(features_mean, additional_feature) 
#     print(f"Haralick features shape: {len(features)}")
#     return features

def haralick_feat_beta(image_path):
    data= cv2.imread(image_path,0)
    print(f"Haralick features shape: {len(data)}")
    return haralick(data).mean(0).tolist()

def haralick_feat_BETA(data):
    data= cv2.imread(data,0)
    print(f"Haralick features shape: {len(data)}")
    return haralick(data).mean(0).tolist()

def bit_glcm_haralick(image_path):
  return bitdesc(image_path) + glcm(image_path) + haralick_feat_beta(image_path)


# def bit_glcm_haralick_beta(data):
    # return bitdesc_beta(data) + glcm_beta(data) + haralick_feat_BETA(data)

def features_extraction_concat(image_path):
    
    rgb_list = []
    try:
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise ValueError("Failed to read image. Please check the image path and format.")
        
        r, g, b = cv2.split(rgb)
        rgb_list.extend([r, g, b, rgb])

        if len(rgb_list) == 4:
            r_features = glcm_beta(rgb_list[0])
            g_features = bitdesc_beta(rgb_list[1])
            b_features = bitdesc_beta(rgb_list[2])
            rgb_features = bit_glcm_haralick(image_path)
            return r_features + g_features + b_features + rgb_features
        else:
            return []
    except Exception as e:
        print(f'Split error: {e}')
        return []


        