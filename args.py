from datetime import datetime
import torch
import argparse
import os

from utils import get_model_attribute

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='arguments')

        # logging & saving
        self.parser.add_argument('--clean_tensorboard', action='store_true', help='Clean tensorboard')
        self.parser.add_argument('--clean_temp', action='store_true', help='Clean temp folder')
        self.parser.add_argument('--log_tensorboard', action='store_true', help='Whether to use tensorboard for logging')
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--print_interval', type=int, default=1, help='loss printing batch interval')
        self.parser.add_argument('--epochs_save', type=int, default=10, help='model save epoch interval')
        self.parser.add_argument('--epochs_validate', type=int, default=1, help='model validate epoch interval')

        # setup
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='cuda:[d] | cpu')
        self.parser.add_argument('--model', default='bert-base-uncased', help='Model type -- bert')
        self.parser.add_argument('--stucture', default='siamese', help='model structure')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to reproduce performance/dataset')

        # Specific to transformers
        self.parser.add_argument('--use_pretrained', default=True, help='use pretrained bert model')
        self.parser.add_argument('--max_len', type=int, default=64, help='maximum sequence length')
        self.parser.add_argument('--pooling', default='mean', help='pooling method to get the sequence embedding: { mean | max | meansqrt| cls }')
        self.parser.add_argument('--do_lower_case', default=True, help='whether to do lower case')
    
        # data construction
        self.parser.add_argument('--data_name', default='financial_news', help='data name')        
        self.parser.add_argument('--phrase_extraction', default='rand_consec3', help='method to extract key phrase')
        self.parser.add_argument('--negative_selection', default='max_like_phrase', help='method to select negative samples')
        self.parser.add_argument('--negative_number', default=2, type=int, help='number of negative samples')
        self.parser.add_argument('--sample_from_same_article', default=True, help='whether to generate negative samples from the same article')
        self.parser.add_argument('--phrase_from_first_sent', default=True, help='whether to generate key phrase from the first sentence')
        

        # training config
        self.parser.add_argument('--batch_size', type=int, default=32, help='batchsize')
        self.parser.add_argument('--epochs', type=int, default=4, help='epochs')
        self.parser.add_argument('--train_steps', type=int, default=50000, help='train steps')
        self.parser.add_argument('--save_checkpoint_steps', type=int, default=40, help='save checkpoint steps')
        self.parser.add_argument('--mode', default='train', help='either train or eval: { train | eval}')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.3, help='Learning rate decay factor')
        self.parser.add_argument('--train_both_branches', default=False, help='Whether to train both branches')

        # Model load parameters
        self.parser.add_argument('--load_model',  default=False, action='store_true', help='whether to load model')
        self.parser.add_argument('--load_model_path', default=None, help='load model path')
        self.parser.add_argument('--load_device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='load device: cuda:[d] | cpu')
        self.parser.add_argument('--epochs_end', type=int, default=3, help='model in which epoch to load')

    def update_args(self):
        args = self.parser.parse_args()
        if args.mode=='eval':
            args.load_model=True
            old_args = args
            fname = os.path.join(args.load_model_path, "model_save", "epoch_{}.dat".format(old_args.epochs_end))
            args = get_model_attribute(
                'saved_args', fname, args.load_device)
            args.mode = 'eval'
            args.device = old_args.load_device
            args.load_model = True
            args.load_model_path = old_args.load_model_path
            args.epochs = old_args.epochs
            args.epochs_end = old_args.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False
            args.produce_graphs = False

            return args

        args.milestones = [args.epochs//5, args.epochs//5*2, args.epochs//5*3, args.epochs//5*4]  # List of milestones

        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        args.fname = args.negative_selection + '_' + args.phrase_extraction + "_" + args.time
        args.dir_input = 'output/'
        args.experiment_path = args.dir_input + args.fname
        args.logging_path = args.experiment_path + '/' + 'logging/'
        args.logging_iter_path = args.logging_path + 'iter_history.csv'
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'
        args.logging_corr_path = args.logging_path + 'corr.csv'
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.tensorboard_path = args.experiment_path + '/' + 'tensorboard/'
        args.dataset_path = 'data/'
        args.temp_path = args.experiment_path + '/' + 'tmp/'

        args.current_model_save_path = args.model_save_path

        args.load_model_path = None

        args.current_dataset_path = None
        args.current_processed_dataset_path = None
        args.current_temp_path = args.temp_path

        return args