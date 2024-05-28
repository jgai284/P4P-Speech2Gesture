# Copyright Chauncey

import h5py
import pandas as pd
import matplotlib.pyplot as plt

def plot_mel_spectrogram(spectrogram, title='Mel Spectrogram', figsize=(10, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def export_to_csv(data, filename='output.csv'):
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(data)
    # Save DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f'Data exported to {filename}')

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        # Check if 'text/meta' is a group or dataset
        if isinstance(file['text/meta'], h5py.Group):
            print("'text/meta' is a group containing:")
            for item in file['text/meta'].keys():
                print(item)  # This will list datasets or groups within 'text/meta'
        elif isinstance(file['text/meta'], h5py.Dataset):
            print("'text/meta' is a dataset.")

def read_h5_data(file_path):
    with h5py.File(file_path, 'r') as file:
        # 读取姿态数据
        pose_data = file['pose/data'][:]
        pose_normalize = file['pose/normalize'][:]
        pose_confidence = file['pose/confidence'][:]
        # 读取音频数据
        audio_log_mel_400 = file['audio/log_mel_400'][:]
        audio_log_mel_512 = file['audio/log_mel_512'][:]
        audio_silence = file['audio/silence'][:]
        #生成csv文件
        # export_to_csv(audio_log_mel_400, 'mel_spectrogram_400.csv')
        plot_mel_spectrogram(audio_log_mel_400)
        # # 读取文本数据
        text_bert = file['text/bert'][:]
        # text_tokens = file['text/tokens'][:]
        # text_w2v = file['text/w2v'][:]
        text_meta_df = pd.read_hdf(file_path, 'text/meta')
        print(text_meta_df.head())

        # 打印或处理数据
        # print("Pose Data:", pose_data)
        # print("pose_normalize", pose_normalize)
        # print("pose_confidence", pose_confidence)
        # print("Audio Log Mel 400 Spectrogram:", audio_log_mel_400)
        # print("Text BERT Embeddings:", text_bert)
        # print("Text Meta DataFrame:", text_meta_df)
        print("text_bert:", text_bert)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path='D:\\UoA\\SOFTENG 700A\\mix-stage-master\\src\\data\\visualization\\cmu0000033570.h5'
    read_h5_data(path)
    # inspect_h5_file(path)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
