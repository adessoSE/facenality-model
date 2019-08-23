# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:43:27 2019

@author: grilborzer
"""
import numpy as np
from preprocessing import data_import as di
from constants import TRAITS, TRAIT_TRESHHOLDS
from math_visualization import extract_trait_values_to_list
from shutil import copyfile, copy

expression = "/neutral/"
path_images = "../dataset/all-cropped" + expression
path_json_train = "../dataset/y_train.json"
path_json_validation = "../dataset/y_validation.json"


def distribute_images_onto_classification_folders(trait_values, y_id_list, train_or_validation):
    
    for i in range(16):
        id_counter = 0
        for value in trait_values[i]:
            copy_image_source_path = "../dataset/all-cropped/neutral/" + str(y_id_list[id_counter]) + ".jpg"
            copy_image_destination_path = "../dataset/classification/" + train_or_validation + "/" + TRAITS[i]
            
            if value >= np.float64(TRAIT_TRESHHOLDS[i]):
                copy_image_destination_path += "/high-range"
            else:
                copy_image_destination_path += "/low-range"
                
            copy(copy_image_source_path, copy_image_destination_path)
            id_counter += 1
    
    
if __name__ == '__main__':   
    y_train, y_train_id_list = di.import_y(path_json_train)
    y_train = np.array(y_train.tolist())
    
    y_validation, y_validation_id_list = di.import_y(path_json_validation)
    y_validation = np.array(y_validation.tolist())
    
    trait_values_train = extract_trait_values_to_list(y_train)
    trait_values_validation = extract_trait_values_to_list(y_validation)
    
    distribute_images_onto_classification_folders(trait_values_train, y_train_id_list, "train")
    distribute_images_onto_classification_folders(trait_values_validation, y_validation_id_list, "validation")