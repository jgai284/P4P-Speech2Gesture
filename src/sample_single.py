from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import h5py
from pycasper.BookKeeper import *
from pycasper.argsUtils import *

# Hide warning message when activate IS metric
import warnings
warnings.filterwarnings('ignore', message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")

def read_log_mel_400(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        log_mel_400_data = h5_file['audio/log_mel_400'][:]
    return log_mel_400_data

def loop(args, exp_num):
    args_subset = ['exp', 'cpk', 'speaker', 'model']
    args_dict_update = {'render':0, 'window_hop':0, 'sample_all_styles':0}
    args_dict_update.update(get_args_update_dict(args))  # Update all the input args

    # Load Args
    book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                      tensorboard=args.tb)
    args = book.args

    # Choose trainer
    Trainer = trainer_chooser(args)

    # Init Trainer
    trainer = Trainer(args, args_subset, args_dict_update)

    # Sample single input
    speech_data_path = 'D:\\UoA\\SOFTENG 700A\\P4P-Speech2Gesture\\src\\data\\visualization\\features\\100912.h5'
    output_path = 'D:\\UoA\\SOFTENG 700A\\P4P-Speech2Gesture\\save\\speech2gesture\\speaker\\test'

    input_spectrogram = read_log_mel_400(speech_data_path)

    trainer.sample_single(input_spectrogram, output_path)

    # Print Experiment No.
    print(args.exp)

if __name__ == '__main__':
    argparseNloop(loop)
