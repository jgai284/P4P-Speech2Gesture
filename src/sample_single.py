from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import pdb
from pycasper.BookKeeper import *
from pycasper.argsUtils import *
import pandas as pd
import numpy as np

# Hide warning message when activate IS metric
import warnings
warnings.filterwarnings('ignore', message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")

def create_csv(file_path):
    # Define the structure of the DataFrame
    data = {
        'dataset': ['train', 'dev', 'test'],
        'delta_time': [np.random.uniform(0, 60) for _ in range(3)],
        'end_time': [pd.to_datetime('2019-06-07 00:13:07') for _ in range(3)],
        'interval_id': ['1', '2', 'live'],
        'speaker': ['oliver'] * 3,
        'start_time': [pd.to_datetime('2019-06-07 00:12:41') for _ in range(3)],
        'video_fn': ['video250'] * 3,
        'video_link': ['http://example.com/video1'] * 3
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a new CSV file
    df.to_csv(file_path, index=False)

def loop(args, exp_num):
  create_csv("E:\\PATS_fake\\pats\\data\\cmu_intervals_df.csv")
  create_csv("E:\\PATS_fake\\pats\\data\\cmu_intervals_df_transforms.csv")

  args_subset = ['exp', 'cpk', 'speaker', 'model']
  args_dict_update = {'render':args.render, 'window_hop':0, 'sample_all_styles':args.sample_all_styles}
  args_dict_update.update(get_args_update_dict(args)) ## update all the input args

  ## Load Args
  book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                    tensorboard=args.tb)
  args = book.args

  ## choose trainer
  Trainer = trainer_chooser(args)

  ## Init Trainer
  trainer = Trainer(args, args_subset, args_dict_update)

  trainer.book._set_seed()
  ## Sample
  trainer.sample_single(exp_num)

  ## Finish exp
  trainer.finish_exp()

  ## Print Experiment No.
  print(args.exp)
  
if __name__ == '__main__':
  argparseNloop(loop)
