# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
import pandas as pd

from transformers import RobertaTokenizer
from ERC_dataset import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader, DACON_loader
from model import ERC_model
from utils import make_batch_roberta, make_batch_bert, make_batch_gpt
from train import evaluate, batch_setting, custom_loss

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support
    

def dacon_test():    
    initial = args.initial
    model_type = args.pretrained    

    make_batch = batch_setting(model_type)

    freeze = args.freeze
    freeze_type = 'freeze' if freeze else 'no_freeze'
    sample = args.sample
    last = True if 'gpt2' in model_type else False
    loss_name = args.loss

    """Dataset Loading"""
    dataset = 'DACON'
    DATA_loader = DACON_loader
    dataclass = args.cls
    dataType = 'multi'


    """Log"""
    log_path = os.path.join('test.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    data_path = os.path.join('dataset', dataset)
    save_path = os.path.join(dataset+'_models', model_type, initial, freeze_type, dataclass, str(sample))
    print("###Save Path### ", save_path)

    dev_path = os.path.join(data_path, dataset+'_dev.txt')
    test_path = os.path.join(data_path, dataset+'_test.txt')

    dev_dataset = DATA_loader(dev_path, dataclass)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)        

    test_dataset = DATA_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    label_list = dev_dataset.labelList
    print('Data: ', dataset, '!!!')
    print('List: ', label_list)
    clsNum = len(label_list)        
    model = ERC_model(model_type, clsNum, last, freeze, initial)
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile))
    model = model.cuda()    
    model.eval()           

    """Dev & Test evaluation"""
    logger.info('####### ' + dataset + ' #######')
    dev_acc, dev_pred_list, dev_label_list, _ = evaluate(loss_name, model, dev_dataloader)
    dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')

    test_acc, test_pred_list, test_label_list, _ = evaluate(loss_name, model, test_dataloader)
    test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')

    logger.info('Fscore ## accuracy: {}, dev-macro: {}, test-macro: {}\n'\
    .format(dev_acc*100, dev_fbeta_macro, test_fbeta_macro))

    test_pred_list = [label_list[tp] for tp in test_pred_list]
    test_csv = pd.DataFrame(test_pred_list, columns=['Target'])
    test_csv.to_csv(os.path.join(save_path, 'test.csv'), index=False)


def main():    
    initial = args.initial
    model_type = args.pretrained
    if 'roberta' in model_type:
        make_batch = make_batch_roberta
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert
    else:
        make_batch = make_batch_gpt      
    freeze = args.freeze
    if freeze:
        freeze_type = 'freeze'
    else:
        freeze_type = 'no_freeze'    
    sample = args.sample
    if 'gpt2' in model_type:
        last = True
    else:
        last = False
    
    """Dataset Loading"""
    dataset_list = ['MELD', 'EMORY', 'iemocap', 'dailydialog']
    DATA_loader_list = [MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader]
    dataclass = args.cls
    dataType = 'multi'
    
    """Log"""
    log_path = os.path.join('test.log')
    fileHandler = logging.FileHandler(log_path)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    for dataset, DATA_loader in zip(dataset_list, DATA_loader_list):
        if dataset == 'MELD':
            data_path = os.path.join('dataset', dataset, dataType)
        else:
            data_path = os.path.join('dataset', dataset)
        save_path = os.path.join(dataset+'_models', model_type, initial, freeze_type, dataclass, str(sample))
        print("###Save Path### ", save_path)
    
        dev_path = os.path.join(data_path, dataset+'_dev.txt')
        test_path = os.path.join(data_path, dataset+'_test.txt')

        dev_dataset = DATA_loader(dev_path, dataclass)
        dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)        

        test_dataset = DATA_loader(test_path, dataclass)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
        print('Data: ', dataset, '!!!')
        clsNum = len(dev_dataset.labelList)        
        model = ERC_model(model_type, clsNum, last, freeze, initial)
        modelfile = os.path.join(save_path, 'model.bin')
        model.load_state_dict(torch.load(modelfile))
        model = model.cuda()    
        model.eval()           

        """Dev & Test evaluation"""
        logger.info('####### ' + dataset + ' #######')
        if dataset == 'dailydialog': # micro & macro
            dev_acc, dev_pred_list, dev_label_list, _ = evaluate(loss_name, model, dev_dataloader)
            dev_pre_macro, dev_rec_macro, dev_fbeta_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
            dev_pre_micro, dev_rec_micro, dev_fbeta_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x

            test_acc, test_pred_list, test_label_list, _ = evaluate(loss_name, model, test_dataloader)
            test_pre_macro, test_rec_macro, test_fbeta_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
            test_pre_micro, test_rec_micro, test_fbeta_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, labels=[0,1,2,3,5,6], average='micro') # neutral x
        else: # weight
            dev_acc, dev_pred_list, dev_label_list, _ = evaluate(loss_name, model, dev_dataloader)
            dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')

            test_acc, test_pred_list, test_label_list, _ = evaluate(loss_name, model, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')

        if dataset == 'dailydialog': # micro & macro
            logger.info('Fscore ## accuracy: {}, dev-macro: {}, dev-micro: {}, test-macro: {}, test-micro: {}'\
                        .format(dev_acc*100, dev_fbeta_macro, dev_fbeta_micro, test_fbeta_macro, test_fbeta_micro))
        else:
            logger.info('Fscore ## accuracy: {}, dev-fscore: {}, test-fscore: {}'.format(test_acc*100, dev_fbeta, test_fbeta))
        logger.info('')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )    
    parser.add_argument( "--pretrained", help = 'roberta-large', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--loss", help = 'ce_loss or focal_loss', default = 'ce_loss')
    parser.add_argument( "--initial", help = 'pretrained or scratch', default = 'pretrained')
    parser.add_argument('-fr', '--freeze', action='store_true', help='freezing PM')
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    dacon_test()
    