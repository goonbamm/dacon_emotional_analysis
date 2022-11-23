
from torch import nn
from transformers import AutoModel


class BaseModel(nn.Module):
    
    def __init__(self, checkpoint, num_classes, dropout=0.5):
        super(BaseModel, self).__init__()

        self.model = AutoModel.from_pretrained(checkpoint)
        self.hiddenDim = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hiddenDim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      return_dict=False)
        
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer