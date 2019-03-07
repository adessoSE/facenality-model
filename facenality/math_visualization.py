import numpy as np
import pandas as pd

def import_data(path):
    y_with_id = pd.read_json(path)
    y = y_with_id.iloc[:, 0].values
    return y, y_with_id


def extract_trait_values_to_list(input_data, list_size=16):
    traits = []
    i = 0
    
    while i < list_size:
        traits.append(input_data[:, i])
        i += 1
    
    return traits


def calculate_variance_and_std(input_data, calculate_whole_list = True, list_size=16, printResult = False):
    variance = []
    std = []
    
    if(calculate_whole_list):
        i = 0
        while i < list_size:
            variance.append(np.var(input_data[i]))
            std.append(np.std(input_data[i]))
            i += 1
    else:
        variance = np.var(input_data)
        std = np.std(input_data)
    
    if(printResult):
        print("variance: ", variance)
        print("std: ", std)
        
    return np.array(variance), np.array(std)

def return_list_rmse_per_trait(prediction, validation, validation_size = 10):
    rmse_per_trait = []
    i = 0

    while i < validation_size:
        rmse_per_trait.append(calculateRMSE(prediction[:, i], validation[:, i]))
        i += 1

    return np.array(rmse_per_trait)


if __name__ == "__main__":
    y, y_with_id = import_data("../dataset/predict.json")
    y = y.tolist()
    y = np.array(y)
    
    traits_to_predict = extract_trait_values_to_list(y)
    traits_average_to_predict = np.average(y, axis=0)
    traits_variance_to_predict, traits_std_to_predict = calculate_variance_and_std(traits_to_predict)
    
    traits_variance_average_to_predict = np.average(traits_variance_to_predict)
    traits_std_average_to_predict = np.average(traits_std_to_predict)
    