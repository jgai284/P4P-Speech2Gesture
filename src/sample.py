from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import pdb
from pycasper.BookKeeper import *
from pycasper.argsUtils import *

# Hide warning message when activate IS metric
import warnings
warnings.filterwarnings('ignore', message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")

def loop(args, exp_num):
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
  trainer.sample(exp_num)

  ## Finish exp
  trainer.finish_exp()

  ## Print Experiment No.
  print(args.exp)
  
if __name__ == '__main__':
  argparseNloop(loop)
