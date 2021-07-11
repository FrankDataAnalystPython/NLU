import config
import torch
import transformers
import torch.nn as nn

class NLU_model(nn.Module):
    def __init__(self,
                 num_int = len(config.INT_encoder.classes_),
                 num_ner = len(config.NER_encoder.classes_)
                 ):
        super(NLU_model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.fc_int = nn.Linear(768, num_int)
        self.fc_ner = nn.Linear(768, num_ner)
        self.num_int = num_int
        self.num_ner = num_ner

    def forward(self, ids, mask, token_type_ids):
        hidden_state, CLS_hidden = self.bert(ids,
                                             attention_mask = mask,
                                             token_type_ids = token_type_ids,
                                             return_dict = False
                                             )
        hidden_state = self.drop1(hidden_state)
        CLS_hidden = self.drop2(CLS_hidden)
        INT_logit = self.fc_int(CLS_hidden)
        NER_logit = self.fc_ner(hidden_state)

        return INT_logit, NER_logit
