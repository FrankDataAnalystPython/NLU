import config
import torch
import pickle

class NLU_dataset:
    def __init__(self,
                 data_in,
                 data_int = None,
                 data_ner = None
                 ):
        self.tokenizer = config.TOKENIZER
        self.INT_encoder = config.INT_encoder
        self.NER_encoder = config.NER_encoder
        self.NER_PAD = config.NER_PAD_IDX

        data_in = [i.replace('\n', '').strip().split() for i in data_in]
        if data_int is not None:
            data_int = [i.replace('\n', '') for i in data_int]
            data_int = config.INT_encoder.transform(data_int).tolist()

        if data_ner is not None:
            data_ner = [i.replace('\n', '').strip().split() for i in data_ner]
            data_ner = [config.NER_encoder.transform(i).tolist() for i in data_ner]

        self.data_in = data_in
        self.data_int = data_int
        self.data_ner = data_ner

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, item):
        # Note the data_in is ['open', 'music' ,'plz']
        data_in = self.data_in[item]
        if self.data_int is not None:
            data_int = self.data_int[item]
        if self.data_ner is not None:
            data_ner = self.data_ner[item]

        # No need to do the intention
        ids = []
        re_data_ner = []

        for i, s in enumerate(data_in):
            inputs = self.tokenizer.encode(s, add_special_tokens = False)
            input_len = len(inputs)
            ids += inputs.ids
            re_data_ner += [data_ner[i]] * input_len

        ids = ids[:config.MAX_LEN - 2]
        re_data_ner = re_data_ner[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        re_data_ner = [self.NER_PAD] + re_data_ner + [self.NER_PAD]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        while len(ids) < config.MAX_LEN:
            ids += [0]
            mask+= [0]
            token_type_ids += [0]
            re_data_ner += [self.NER_PAD]

        if (self.data_int is not None) and (self.data_ner is not None):
            return {'ids' : torch.tensor(ids, dtype = torch.long),
                    'mask': torch.tensor(mask, dtype = torch.long),
                    'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                    'NER_target' : torch.tensor(re_data_ner, dtype = torch.long),
                    'INT_target' : torch.tensor(data_int)
                    }
        else:
            return {'ids' : torch.tensor([ids], dtype = torch.long),
                    'mask': torch.tensor([mask], dtype = torch.long),
                    'token_type_ids' : torch.tensor([token_type_ids], dtype = torch.long),
                    'NER_target' : None,
                    'INT_target' : None
                    }

if __name__ == '__main__':
    with open(config.TRAIN_FILE_X, 'r') as f:
        train_in = f.readlines()
    with open(config.TRAIN_FILE_INT, 'r') as f:
        train_int = f.readlines()
    with open(config.TRAIN_FILE_NER, 'r') as f:
        train_ner = f.readlines()
    ND = NLU_dataset(train_in, train_int, train_ner)
    d = ND[0]

    print('end')










