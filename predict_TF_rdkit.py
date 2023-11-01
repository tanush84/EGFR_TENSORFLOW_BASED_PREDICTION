import pandas as pd
import numpy as np
import matplotlib as mlt
import csv 
import os
import operator
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import argparse

df = []
tf.keras.backend.clear_session()
# Set the environment variable to disable GPU. comment out this line is trying to make prediction using gpu.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def read_smi_file(file_path):
    try:
        with open(file_path, 'r') as smi_file:
            lines = smi_file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def smi_to_dataframe(file_path):
    smi_data = read_smi_file(file_path)
    if not smi_data:
        return None
    
    data = {'SMILES': smi_data}
    df = pd.DataFrame(data)
    return df

def calculate_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return list(fingerprint)
    else:
        print(f"Unable to parse SMILES: {smiles}")
        return [0] * 2048


def main():
    parser = argparse.ArgumentParser(description="Convert a .smi file to a pandas DataFrame.")
    parser.add_argument('input_file', type=str, help='Path to the .smi file')

    args = parser.parse_args()
    input_file = args.input_file

    df = smi_to_dataframe(input_file)

    if df is not None:
        print("DataFrame created successfully:")
        print(df.head())
        with tf.device('/CPU:0'):                          ### use /gpu:id   if want to use gpu
            dftest = pd.read_csv(input_file, delimiter='\t', header=None, names=["SMILES"])
            dftest['MorganFingerprint'] = dftest['SMILES'].apply(calculate_morgan_fingerprint)
            df_fingerprints = dftest['MorganFingerprint'].apply(pd.Series)
            # Concatenate the new columns with the original DataFrame
            dftest = pd.concat([dftest, df_fingerprints], axis=1)
            # Drop the original "MorganFingerprint" column
            dftest = dftest.drop(columns=['MorganFingerprint'])
            dftest_pred = dftest.drop(dftest.columns[[0]], axis=1)
     
            # load the model from disk
            loaded_model = keras.models.load_model('./assets/tfmodel-egfr.h5')

            result_pred = loaded_model.predict(dftest_pred)
        
            # Apply a threshold of 0.5 to obtain binary predictions
            binary_pred = (result_pred >= 0.5).astype(int)

            # binary_predictions will be a NumPy array with 0s and 1s
            if binary_pred == 0:
                print('Molecule is inactive')
            else:
                print('Molecule is Active')

    else:
        print("DataFrame creation failed.")
    
    


if __name__ == "__main__":
    main()



