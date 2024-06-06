import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import nltk
from transformers import BertTokenizer
from data.dataUtils import Data  # Assuming your main code is in data_loading_module.py

def main():
    # # Initialize the Data class
    # path2data = '/path/to/your/data'  # Update this path to where your data is located
    # speaker = 'all'  # You can specify specific speakers if needed
    # modalities = ['pose/data', 'audio/log_mel_512']
    # fs_new = [15, 15]
    # time = 4.3
    # batch_size = 4
    # shuffle = True
    # num_workers = 0

    # data_instance = Data(path2data=path2data, 
    #                      speaker=speaker, 
    #                      modalities=modalities, 
    #                      fs_new=fs_new, 
    #                      time=time, 
    #                      batch_size=batch_size, 
    #                      shuffle=shuffle, 
    #                      num_workers=num_workers)

    # # Load data
    # data_loader = data_instance.dataLoader_train

    # # Fetch one batch of data
    # for batch in data_loader:
    #     print(batch)
    #     break

    print("hihi")

if __name__ == "__main__":
    main()
