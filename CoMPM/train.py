# -*- coding: utf-8 -*-
import os
import pdb
import torch
import wandb
import random
import logging
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm.auto import tqdm
from model import ERC_model
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from loss import FocalLoss
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from ERC_dataset import DACON_loader, MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def custom_loss(loss_name, pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """

    if loss_name == 'ce_loss':
        loss = nn.CrossEntropyLoss()
    
    elif loss_name == 'focal_loss':
        loss = FocalLoss()
    
    loss_val = loss(pred_outs, labels)
    
    return loss_val


def dataset_setting(dataset, dyadic):
    data_type = 'multi'

    if dataset == 'MELD':
        if dyadic:
            data_type = 'dyadic'

        data_path = './dataset/MELD/'+data_type+'/'
        DATA_loader = MELD_loader

    elif dataset == 'EMORY':
        data_path = './dataset/EMORY/'
        DATA_loader = Emory_loader

    elif dataset == 'iemocap':
        data_path = './dataset/iemocap/'
        DATA_loader = IEMOCAP_loader

    elif dataset == 'dailydialog':
        data_path = './dataset/dailydialog/'
        DATA_loader = DD_loader    

    elif dataset == 'DACON':
        data_path = './dataset/DACON/'
        DATA_loader = DACON_loader
    
    else:
        print('DATASET NAME IS WRONG!!')
        exit()
        
    return data_path, DATA_loader


def batch_setting(model_type):
    if 'berta' in model_type:
        make_batch = make_batch_roberta
        
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert

    else:
        make_batch = make_batch_gpt
        
    return make_batch


def get_dataloaders(data_path, dataset, DATA_loader, dataclass, batch_size, make_batch, sample):
    # dataset loading
    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'
            
    train_dataset = DATA_loader(train_path, dataclass)

    if sample < 1.0:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=make_batch)

    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)

    train_sample_num = int(len(train_dataloader) * sample)
    
    dev_dataset = DATA_loader(dev_path, dataclass)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    test_dataset = DATA_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    dataloaders = (train_dataloader, dev_dataloader, test_dataloader)
    info = (train_dataset.labelList, len(train_dataset), train_sample_num)

    return dataloaders, info


def logger_setting(dataset, model_type, initial, freeze, dataclass, sample):
    save_path = os.path.join(dataset+'_models', model_type, initial,
                             'freeze' if freeze else 'no_freeze', dataclass, str(sample))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("###Save Path### ", save_path)

    log_path = os.path.join(save_path, 'train.log')
    fileHandler = logging.FileHandler(log_path)
    streamHandler = logging.StreamHandler()
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)

    return save_path


def main():
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # hyperparameter setting
    dataset = args.dataset
    dataclass = args.cls
    batch_size = args.batch
    sample = args.sample
    model_type = args.pretrained
    model_name = model_type.split('/')[-1]
    freeze = args.freeze
    initial = args.initial
    early_stop = args.earlystop
    use_amp = args.amp
    loss_name = args.loss
    training_epochs = args.epoch
    max_grad_norm = args.norm
    lr = args.lr
    use_wandb = args.wandb
    seed = 2022
    
    # seed setting
    seed_everything(seed=2022)
    
    CONFIG = {
        # dataset
        'dataset': dataset,
        'dataclass': dataclass,
        'batch_size': batch_size,
        'sample': sample,
        # model
        'model_type': model_type,
        'freeze': freeze,
        'initial': initial,
        # training
        'early_stop': early_stop,
        'use_amp': use_amp,
        'epochs': training_epochs,
        'max_grad_norm': max_grad_norm,
        'learing_rate': lr,
        'loss_name': loss_name,
        'seed': seed,
    }
 
    # get dataloaders   
    data_path, DATA_loader = dataset_setting(dataset, args.dyadic)    
    make_batch = batch_setting(model_type)

    dataloaders, info = get_dataloaders(data_path, dataset, DATA_loader,
                                        dataclass, batch_size, make_batch, sample)

    train_dataloader, dev_dataloader, test_dataloader = dataloaders    
    label_list, num_warmup_steps, train_sample_num = info
    num_training_steps = num_warmup_steps * training_epochs    
 
    # logging and path
    save_path = logger_setting(dataset, model_type, initial, freeze, dataclass, sample)
    
    # model
    last = True if 'gpt2' in model_type else False
    model = ERC_model(model_type, len(label_list), last, freeze, initial)
    model = model.cuda()
    model.train() 
    
    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.train_params, lr=lr) # , eps=1e-06, weight_decay=0.01

    if not use_amp:
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
    
    # Input & Label Setting
    best_dev_fscore, best_test_fscore = 0, 0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro = 0, 0, 0, 0    
    best_epoch = 0
    patience = 0 # for early stop
    
    scaler = GradScaler(enabled=use_amp)
    
    # wandb setting
    if use_wandb:
        wandb.init(project='dacon_sentiment_analysis', config=CONFIG)
    
    for epoch in tqdm(range(training_epochs), desc='training loops'):
        model.train() 
        train_loss = []
        
        for i_batch, data in tqdm(enumerate(train_dataloader), desc='batch'):
            optimizer.zero_grad()
            
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_logits = model(batch_input_tokens, batch_speaker_tokens)
                batch_loss = custom_loss(loss_name, pred_logits, batch_labels)

            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            scaler.step(optimizer)
            scaler.update()

            if not use_amp:
                scheduler.step()
            
            train_loss.append(batch_loss.item())
        
        train_loss = np.mean(train_loss)
        
        """Dev & Test evaluation"""
        model.eval()

        if dataset == 'dailydialog': # micro & macro
            dev_acc, dev_pred_list, dev_label_list, dev_loss = evaluate(loss_name, model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
            
            dev_fscore = dev_fbeta_macro+dev_fbeta_micro

            """Best Score & Model Save"""
            if dev_fscore > best_dev_fscore_macro + best_dev_fscore_micro:
                best_dev_fscore_macro = dev_fbeta_macro                
                best_dev_fscore_micro = dev_fbeta_micro
                
                test_acc, test_pred_list, test_label_list, test_loss = evaluate(loss_name, model, test_dataloader)
                test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
                test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
                
                best_epoch = epoch
                save_model(model, save_path)
                
                patience = 0
                
            else:
                patience += 1

        elif dataset == 'DACON': # macro
            dev_acc, dev_pred_list, dev_label_list, dev_loss = evaluate(loss_name, model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            
            dev_fscore = dev_fbeta_macro

            """Best Score & Model Save"""
            if dev_fscore > best_dev_fscore_macro:
                best_dev_fscore_macro = dev_fbeta_macro

                test_acc, test_pred_list, test_label_list, test_loss = evaluate(loss_name, model, test_dataloader)
                test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')

                test_pred_list = [label_list[tp] for tp in test_pred_list]
                test_csv = pd.DataFrame(test_pred_list, columns=['Target'])
                test_csv.to_csv(f'./test_epoch_{epoch}_on_{model_name}.csv', index=False)

                best_epoch = epoch
                save_model(model, save_path)

                patience = 0
                
            else:
                patience += 1

        else: # weight
            dev_acc, dev_pred_list, dev_label_list, dev_loss = evaluate(loss_name, model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            """Best Score & Model Save"""
            if dev_fbeta > best_dev_fscore:
                best_dev_fscore = dev_fbeta
                
                test_acc, test_pred_list, test_label_list, test_loss = evaluate(loss_name, model, test_dataloader)
                test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
                
                test_csv = pd.DataFrame(test_pred_list, columns=['Target'])
                test_csv.to_csv('./best_test.csv', index=False)
                
                best_epoch = epoch
                save_model(model, save_path)

                patience = 0
            
            else:
                patience += 1
        
        logger.info('Epoch: {}'.format(epoch))
        
        if dataset == 'dailydialog': # micro & macro
            logger.info('Development ## accuracy: {}, macro-fscore: {}, micro-fscore: {}'.format(dev_acc, dev_fbeta_macro, dev_fbeta_micro))
            logger.info('')
            
        elif dataset == 'DACON':
            logger.info('Development ## train_loss: {}, valid_loss: {}, macro-fscore: {}'.format(train_loss, dev_loss, dev_fbeta_macro))
            logger.info('')
            if use_wandb:
                wandb.log({'train_loss': train_loss, 'valid_loss': dev_loss, 'valid_f1_score': dev_fbeta_macro})            
            
        else:
            logger.info('Development ## accuracy: {}, precision: {}, recall: {}, fscore: {}'.format(dev_acc, dev_pre, dev_rec, dev_fbeta))
            logger.info('')
        
        if patience == early_stop:
            logger.info('#### Early Stop ####')
            logger.info('')
            break
        
    if dataset == 'dailydialog': # micro & macro
        logger.info('Final Fscore ## test-accuracy: {}, test-macro: {}, test-micro: {}, test_epoch: {}'.format(test_acc, test_fbeta_macro, test_fbeta_micro, best_epoch))
    
    elif dataset != 'DACON':
        logger.info('Final Fscore ## test-accuracy: {}, test-fscore: {}, test_epoch: {}'.format(test_acc, test_fbeta, best_epoch))         
   
    
def evaluate(loss_name, model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    loss = []
    
    # label arragne
    with torch.no_grad():
        for data in dataloader:            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_speaker_tokens = data
            batch_input_tokens, batch_labels = batch_input_tokens.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_speaker_tokens) # (1, clsNum)
            batch_loss = custom_loss(loss_name, pred_logits, batch_labels)
            
            """Calculation"""    
            pred_label = pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            
            if pred_label == true_label:
                correct += 1
            
            loss.append(batch_loss.item())
            
        acc = correct/len(dataloader)
        
    return acc, pred_list, label_list, np.mean(loss)


def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    # parameters
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--wandb", action='store_true', help='use_wandb')

    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog or DACON', default='MELD')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--sample", type=float, help = "sampling training dataset", default=1.0)

    parser.add_argument( "--batch", type=int, help = "batch_size", default=1)    
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default=100) # 12 for iemocap
    parser.add_argument( "--lr", type=float, help = "learning rate", default=1e-6)
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default=10)
    parser.add_argument( "--earlystop", type=int, help = "early stop", default=3)
    parser.add_argument( "--amp", action='store_true', help = "Auto Mixed Precision")
    parser.add_argument( "--loss", help = 'ce_loss or focal_loss', default = 'ce_loss')

    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument( "--pretrained", help = 'roberta-base/roberta-large or bert-large-uncased or gpt2 or gpt2-large or gpt2-medium', default = 'roberta-base')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
        
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    
    main()
        