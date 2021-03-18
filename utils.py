import os
import shutil
import pickle
import torch
import numpy as np


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot/(norm1*norm2)
    

def save_model(epoch, args, model, optimizer=None, scheduler=None, **extra_args):
    if not os.path.isdir(args.current_model_save_path):
        os.makedirs(args.current_model_save_path)

    fname = args.current_model_save_path +'epoch' + '_' + str(epoch) + '.dat'
    checkpoint = {'saved_args': args, 'epoch': epoch}

    save_items = {'model': model}

    if optimizer:
        save_items['optimizer'] = optimizer
    if scheduler:
        save_items['scheduler'] = scheduler


    for name, d in save_items.items():
        save_dict = d.state_dict()
        checkpoint[name] = save_dict

    if extra_args:
        for arg_name, arg in extra_args.items():
            checkpoint[arg_name] = arg

    torch.save(checkpoint, fname)


def load_model(path, device, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location=device)

    for name, d in {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}.items():
        if d is not None:
            d.load_state_dict(checkpoint[name])

        if name == 'model':
            d.to(device=device)


def get_last_checkpoint(args, epoch):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = args.load_model_path + '/model_save/'
    # Checkpoint file names are in lexicographic order
    last_checkpoint_name = checkpoint_dir + 'epoch' + '_' + str(epoch) + '.dat'
    print('Last checkpoint is {}'.format(last_checkpoint_name))
    return last_checkpoint_name, epoch


def get_model_attribute(attribute, fname, device):

    checkpoint = torch.load(fname, map_location=device)

    return checkpoint[attribute]


# Create Directories for outputs
def create_dirs(args):
    if args.clean_tensorboard and os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)

    if args.clean_temp and os.path.isdir(args.temp_path):
        shutil.rmtree(args.temp_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    if not os.path.isdir(args.temp_path):
        os.makedirs(args.temp_path)

    if not os.path.isdir(args.tensorboard_path):
        os.makedirs(args.tensorboard_path)

    if not os.path.isdir(args.current_temp_path):
        os.makedirs(args.current_temp_path)

    if not os.path.isdir(args.logging_path):
        os.makedirs(args.logging_path)