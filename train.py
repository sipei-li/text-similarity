from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AdamW
import time
import datetime
from collections import defaultdict
import os
import pandas as pd
import numpy as np

from utils import save_model, load_model, get_model_attribute, get_last_checkpoint
# from models import data_loader 
# from models.data_loader import load_dataset


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_epoch(
        epoch, args, model, train_dataloader,
        optimizer, scheduler, log_history, summary_writer=None):
    # Set training mode for modules
    model.train()

    # batch_count = len(train_iter)
    total_train_loss = 0.0
    total_train_accuracy = 0
    for batch_id, batch in enumerate(train_dataloader):

        model.zero_grad()

        st = time.time()

        b_input_ids1 = batch[0].to(args.device)
        b_input_mask1 = batch[1].to(args.device)
        b_input_ids2 = batch[2].to(args.device)
        b_input_mask2 = batch[3].to(args.device)
        b_labels = batch[4].to(args.device)
        features1 = {'input_ids':b_input_ids1, 'attention_mask':b_input_mask1}
        features2 = {'input_ids':b_input_ids2, 'attention_mask':b_input_mask2}

        loss, logits = model(features1,
                             features2,
                             labels=b_labels)


        total_train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        acc = flat_accuracy(logits, label_ids)
        total_train_accuracy += acc

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        spent = time.time() - st
        if batch_id % args.print_interval == 0:
            print('epoch {} batch {}: loss is {}, accuracy is {}, time spent is {}.'.format(epoch, batch_id, loss, acc, spent), flush=True)

        log_history['batch_loss'].append(loss.item())
        log_history['batch_time'].append(spent)

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.negative_selection, args.phrase_extraction), loss.item(), batch_id)

    return total_train_loss / (batch_id+1), total_train_accuracy / (batch_id+1)


def test_data(epoch, args, model, eval_dataloader):
    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch_id, batch in enumerate(eval_dataloader):
        b_input_ids1 = batch[0].to(args.device)
        b_input_mask1 = batch[1].to(args.device)
        b_input_ids2 = batch[2].to(args.device)
        b_input_mask2 = batch[3].to(args.device)
        b_labels = batch[4].to(args.device)
        features1 = {'input_ids':b_input_ids1, 'attention_mask':b_input_mask1}
        features2 = {'input_ids':b_input_ids2, 'attention_mask':b_input_mask2}
        with torch.no_grad():        
            (loss, logits) = model(features1,
                                   features2,
                                   labels=b_labels)

        total_eval_loss += loss.item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    avg_val_accuracy = total_eval_accuracy / (batch_id+1)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / (batch_id+1)

    return total_eval_loss / (batch_id+1), avg_val_accuracy


def train(args, model, train_dataloader, eval_dataloader):
    total_t0 = time.time()
    num_training_steps = args.epochs * args.batch_size
    optimizer = AdamW(model.parameters(), args.lr, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = num_training_steps)
    
    log_history = defaultdict(list)
    if args.load_model:
        fname, epoch = get_last_checkpoint(args, epoch=args.epochs_end)
        load_model(path=fname, device=args.device, model=model, optimizer=optimizer, scheduler=scheduler)
        print('Model loaded')
        df_iter = pd.read_csv(os.path.join(args.logging_iter_path))
        df_epoch = pd.read_csv(os.path.join(args.logging_epoch_path))

        log_history['batch_loss'] = df_iter['batch_loss'].tolist()
        log_history['batch_time'] = df_iter['batch_time'].tolist()
        log_history['train_loss'] = df_epoch['train_loss'].tolist()
        log_history['valid_loss'] = df_epoch['valid_loss'].tolist()

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path+ ' ' + args.time, flush_secs=5)
    else:
        writer = None

    # train_iter = iter_fct(args, shuffle=True, is_test=False, corpus_type='train')
    # eval_iter = iter_fct(args, shuffle=False, is_test=False, corpus_type='valid')
    for epoch in range(args.epochs):
        #train
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        print('Training...')
        t0 = time.time()
        loss, acc = train_epoch(
            epoch, args, model, train_dataloader, optimizer, scheduler, log_history, writer)
        # train_iter = iter_fct(args, shuffle=True, is_test=False, corpus_type='train')
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(loss))
        print("  Average training accuracy: {0:.2f}".format(acc))
        print("  Training epcoh took: {:}".format(training_time))

        #evaluate
        print("")
        print("Running Validation...")

        t0 = time.time()
        avg_val_loss, avg_val_accuracy = test_data(epoch, args, model, eval_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print("  Validation took: {:}".format(validation_time))

        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/validate'.format(args.negative_selection, args.phrase_extraction), avg_val_loss, avg_val_accuracy, epoch)
        
        # save model
        save_model(epoch, args, model, optimizer, scheduler)

        print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        log_history['train_loss'].append(loss)
        log_history['valid_loss'].append(avg_val_loss)

        # save logging history
        df_iter = pd.DataFrame()
        df_epoch = pd.DataFrame()
        df_iter['batch_loss'] = log_history['batch_loss']
        df_iter['batch_time'] = log_history['batch_time']
        df_epoch['train_loss'] = log_history['train_loss']
        df_epoch['valid_loss'] = log_history['valid_loss']

        df_iter.to_csv(args.logging_iter_path, index=False)
        df_epoch.to_csv(args.logging_epoch_path, index=False)

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

