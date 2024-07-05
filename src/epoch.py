# This file is responsible for validating overfitting and underfitting problems by providing loss against epoch

import json
import pandas as pd
import matplotlib.pyplot as plt

def inspect_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    def print_json_structure(d, indent=0):
        for key, value in d.items():
            print('  ' * indent + str(key) + ': ' + str(type(value)))
            if isinstance(value, dict):
                print_json_structure(value, indent + 1)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # Print structure of first dict in list for simplicity
                print('  ' * (indent + 1) + '[0]: ' + str(type(value[0])))
                print_json_structure(value[0], indent + 2)

    # Print the structure of the JSON file
    if isinstance(data, dict):
        print_json_structure(data)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f'[{i}]: ' + str(type(item)))
            if isinstance(item, dict):
                print_json_structure(item, 1)
    else:
        print('The JSON file contains an unsupported structure.')

def plot_loss(file_path):
    # Read the JSON files
    df_400 = pd.read_json(file_path)
    df_512 = pd.read_json("D:\\UoA\SOFTENG 700A\\P4P-Speech2Gesture\\save\\speech2gesture\\oliver\\exp_114_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_res.json")
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))  # 3 rows, 2 columns
    
    # Plot for subfigure 1: L1 Loss (column 1)
    axs[0, 0].plot(df_400['test_L1'], label='Test L1 Loss (400)')
    axs[0, 0].plot(df_400['train_L1'], label='Train L1 Loss (400)')
    axs[0, 0].set_title('L1 Loss (400)')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].legend()

    axs[0, 1].plot(df_512['test_L1'], label='Test L1 Loss (512)')
    axs[0, 1].plot(df_512['train_L1'], label='Train L1 Loss (512)')
    axs[0, 1].set_title('L1 Loss (512)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].legend()
    
    # Plot for subfigure 2: F1 Score (column 1)
    axs[1, 0].plot(df_400['test_F1'], label='Test F1 Score (400)')
    axs[1, 0].plot(df_400['train_F1'], label='Train F1 Score (400)')
    axs[1, 0].set_title('F1 Score (400)')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].legend()

    axs[1, 1].plot(df_512['test_F1'], label='Test F1 Score (512)')
    axs[1, 1].plot(df_512['train_F1'], label='Train F1 Score (512)')
    axs[1, 1].set_title('F1 Score (512)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].legend()
    
    # Plot for subfigure 3: FID Loss (column 1)
    axs[2, 0].plot(df_400['test_FID'], label='Test FID Score (400)')
    axs[2, 0].plot(df_400['train_FID'], label='Train FID Score (400)')
    axs[2, 0].set_title('FID Score (400)')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Loss')
    axs[2, 0].legend()

    axs[2, 1].plot(df_512['test_FID'], label='Test FID Score (512)')
    axs[2, 1].plot(df_512['train_FID'], label='Train FID Score (512)')
    axs[2, 1].set_title('FID Score (512)')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].set_ylabel('Loss')
    axs[2, 1].legend()
    
    # Adjust layout and spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    file_path = "D:\\UoA\SOFTENG 700A\\P4P-Speech2Gesture\\save\\speech2gesture\\oliver\\exp_113_cpk_speech2gesture_speaker_['oliver']_model_Speech2Gesture_G_note_speech2gesture_res.json"
    # inspect_json(file_path)
    plot_loss(file_path)