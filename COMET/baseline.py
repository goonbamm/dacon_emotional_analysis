import os
import torch
import wandb
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

from utils import seed_everything, competition_metric, drop_columns_by_none_ratio
from dataset import DaconDataset
from model import BaseModel


def train(CONFIG, model, optimizer, scheduler, train_loader, valid_loader, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler(enabled=CONFIG['USE_AMP'])

    best_score = 0
    best_model_path = os.path.join(data_path, 'best_model')

    patience = 0

    for epoch_num in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = []

        for input_ids, attention_mask, train_label in tqdm(train_loader):
            optimizer.zero_grad()

            train_label = train_label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            with torch.cuda.amp.autocast(enabled=CONFIG['USE_AMP']):
              output = model(input_id, mask)
              batch_loss = criterion(output, train_label.long())

            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['MAX_NORM'])
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            train_loss.append(batch_loss.item())

        valid_loss, valid_score = validation(model, criterion, valid_loader, device)
        print(f'Epoch [{epoch_num}], Train Loss : [{np.mean(train_loss) :.5f}] Valid Loss : [{np.mean(valid_loss) :.5f}] Valid F1 Score : [{valid_score:.5f}]')
        wandb.log({'train_loss': np.mean(train_loss), 'valid_loss': np.mean(valid_loss), 'valid_f1_score': valid_score})

        if best_score < valid_score:
            print(f'best F1 Score: {best_score} → {valid_score}')

            best_score = valid_score
            patience = 0

            # save best model
            if os.path.exists(best_model_path):
              shutil.rmtree(best_model_path) # delete everything in the directory

            model.save_pretrained(best_model_path)

        else:
          patience += 1

          if patience == CONFIG['EARLY_STOP']:
            break
        
    return best_model


def validation(model, criterion, test_loader, device):
    model.eval()

    val_loss = list()
    model_preds = list()
    true_labels = list()

    with torch.no_grad():
        for input_ids, attention_mask, valid_label in tqdm(test_loader):
            valid_label = valid_label.to(device)
            input_id = input_ids.to(device)
            mask = attention_mask.to(device)

            output = model(input_id, mask)
    
            batch_loss = criterion(output, valid_label.long()) 
            val_loss.append(batch_loss.item())      
            
            model_preds += output.argmax(1).detach().cpu().numpy().tolist()
            true_labels += valid_label.detach().cpu().numpy().tolist()
            
        val_f1 = competition_metric(true_labels, model_preds)

    return val_loss, val_f1    


def inference(data_path, infer_model, le, tokenizer, device, relation_name):
    infer_model.eval()

    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test = DaconDataset(test, tokenizer, mode ='test')
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    infer_model.to(device)
    infer_model.eval()
    
    test_predict = []
    
    for input_ids, attention_mask in tqdm(test_dataloader):
        input_id = input_ids.to(device)
        mask = attention_mask.to(device)
        y_pred = infer_model(input_id, mask)
        test_predict += y_pred.argmax(1).detach().cpu().numpy().tolist()

    preds = le.inverse_transform(test_predict)

    submit = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    submit['Target'] = preds
    submit.head()
    submit.to_csv(f'./result/{relation_name}.csv', index=False)


def main(CONFIG, data_path, best_model_path, relation_num):
    # device setting
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device', device)
    
    # seed setting
    seed_everything(CONFIG['SEED'])

    # dataset setting    
    data_csv = pd.read_csv(os.path.join(data_path, 'train.csv'))
    data_csv, drop_list = drop_columns_by_none_ratio(data_csv, threshold=0.3)
    relation_name = data_csv.columns[4 + relation_num] # 0 ~ 4: ID, Utterance, Speaker, Dialogue_ID, Target
    
    le = LabelEncoder()
    le = le.fit(data_csv['Target'])
    data_csv['Target'] = le.transform(data_csv['Target'])

    train_csv = data_csv[~data_csv['Dialogue_ID'].isin([i for i in range(1016,1039)])].reset_index(drop=True)
    valid_csv = data_csv[data_csv['Dialogue_ID'].isin([i for i in range(1016,1039)])].reset_index(drop=True)

    print(f'train_dataset: {len(train_csv)}')
    print(f'valid_dataset: {len(valid_csv)}')
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['CHECKPOINT'])
    
    train_dataset = DaconDataset(train_csv, relation_name, tokenizer, mode='train')
    valid_dataset = DaconDataset(valid_csv, relation_name, tokenizer, mode='valid')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    # model setting
    model = BaseModel(checkpoint=CONFIG['CHECKPOINT'], num_classes=len(le.classes_))

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = CONFIG["LEARNING_RATE"])
    num_training_steps = CONFIG['EPOCHS'] * len(train_dataloader)
    lr_scheduler = get_scheduler(name=CONFIG['SCHEDULER_NAME'], optimizer=optimizer,
                                num_warmup_steps=CONFIG['NUM_WARMUP_STEPS'],
                                num_training_steps=num_training_steps)


    wandb.init(project='dacon_sentiment_analysis', name='{}_relation: {}'.format(CONFIG['CHECKPOINT'], relation_name))

    infer_model = train(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, device)
    wandb.finish()
    
    """
    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)

    torch.save(infer_model.state_dict(), os.path.join(best_model_path, 'baseline.pt'))
    """
    
    inference(data_path, infer_model, le, tokenizer, device, relation_name)
    
    del(infer_model)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    CONFIG = {
        # seed
        'SEED': 2022,

        # model
        'CHECKPOINT': 'tae898/emoberta-large',
        'LEARNING_RATE': 1e-6,

        # scheduler
        'SCHEDULER_NAME': 'linear',
        'NUM_WARMUP_STEPS': 0,

        # training
        'EPOCHS': 100,
        'BATCH_SIZE': 16,
        'EARLY_STOP': 3,
        'USE_AMP': True,
        'MAX_NORM': 5,
    }

    data_path = './dataset'
    best_model_path = './model_weight'

    relation_num = 1
    main(CONFIG=CONFIG, data_path=data_path, best_model_path=best_model_path, relation_num=relation_num)