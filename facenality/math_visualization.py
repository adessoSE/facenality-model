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


if __name__ == "__main__":
    y, y_with_id = import_data("../dataset/all.json")
    y = y.tolist()
    y = np.array(y)
    
    traits = extract_trait_values_to_list(y)
    traits_average = np.average(y, axis=0)
    traits_variance, traits_std = calculate_variance_and_std(traits)
    
    traits_variance_average = np.average(traits_variance)
    traits_std_average = np.average(traits_std)