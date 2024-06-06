import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import nltk
from transformers import BertTokenizer
from data.dataUtils import Data

def main():
    # Specify the path storing cmu_intervals_df.csv and cmu_intervals_df_transform.csv
    # E.g. 'UoA/James' gives 'UoA/James/cmu_intervals_df.csv'

    # path2data = 'D:/UoA/SOFTENG 700A/P4P-Speech2Gesture'
    path2data = 'F:/PATSDATASET/oliver/pats/data'
    speaker = ["oliver"]
    modalities = ['pose/data', 'audio/log_mel_512']
    fs_new = [15, 15]
    time = 4.3
    batch_size = 4
    shuffle = True
    num_workers = 0

    data_instance = Data(path2data=path2data, 
                         speaker=speaker, 
                         modalities=modalities, 
                         fs_new=fs_new, 
                         time=time, 
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         num_workers=num_workers)

    # Load training data (e.g. ['train'] ['dev'] ['test'])
    data_loader = data_instance.train

    # Fetch one batch of data
    for batch in data_loader:
        print(batch)
        break

    # print("hihi")

if __name__ == "__main__":
    main()
