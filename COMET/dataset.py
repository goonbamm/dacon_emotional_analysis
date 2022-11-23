from torch.utils.data import DataLoader, Dataset


class DaconDataset(Dataset):      
    def __init__(self, data, relation_name, tokenizer, mode='train'):
        self.dataset = data
        self.r_name = relation_name
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        text = self.dataset['Utterance'][idx]
        comet_text = f' {self.r_name}: ' + str(self.dataset[self.r_name][idx])
        inputs = self.tokenizer(text, comet_text, padding='max_length', max_length=512,
                                truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
    
        if self.mode in ['train', 'valid']:
            y = self.dataset['Target'][idx]
            return input_ids, attention_mask, y

        else:
            return input_ids, attention_mask