import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model import *
import pickle

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def loss_fn_NER(NER_logit, NER_target):
    NER_logit = NER_logit.reshape(-1, NER_logit.shape[-1])
    NER_target= NER_target.reshape(-1)
    lfn = nn.CrossEntropyLoss(ignore_index = config.NER_PAD_IDX,
                              reduction = 'sum')
    loss = lfn(NER_logit, NER_target)
    return loss

def loss_fn_INT(INT_logit, INT_target):
    lfn = nn.CrossEntropyLoss(reduction='sum')
    loss = lfn(INT_logit, INT_target)
    return loss

def total_loss(INT_logit, INT_target, NER_logit, NER_target,
               ratio = len(config.NER_encoder.classes_) / len(config.INT_encoder.classes_)):
    loss_int = loss_fn_INT(INT_logit, INT_target)
    loss_ner = loss_fn_NER(NER_logit, NER_target)
    return ratio * loss_int + loss_ner


def train(model, iterator, optimizer, device, scheduler, epoch_tuple):
    epoch_loss = 0
    total_len = 0
    model.train()
    TQDM = tqdm(enumerate(iterator), total = len(iterator), leave = False)
    for idx, batch in TQDM:
        for key in batch:
            batch[key] = batch[key].to(device, dtype = torch.long)
        ids = batch['ids']
        mask= batch['mask']
        token_type_ids = batch['token_type_ids']
        NER_target = batch['NER_target']
        INT_target = batch['INT_target']
        optimizer.zero_grad()

        INT_logit, NER_logit = model(ids, mask, token_type_ids)
        loss = total_loss(INT_logit, INT_target, NER_logit, NER_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        total_len += 1
        TQDM.set_description(f'Epoch [{epoch_tuple[0]}/{epoch_tuple[1]}]')
        TQDM.set_postfix({'loss': epoch_loss / total_len,
                          #'bleu': epoch_bleu / total_len,
                          # 'acc_pos': epoch_acc_pos / total_len
                          })
    return epoch_loss / total_len

def evaluate(model, iterator, device):
    epoch_loss = 0
    total_len = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            for key in batch:
                batch[key] = batch[key].to(device, dtype=torch.long)
            ids = batch['ids']
            mask = batch['mask']
            token_type_ids = batch['token_type_ids']
            NER_target = batch['NER_target']
            INT_target = batch['INT_target']

            INT_logit, NER_logit = model(ids, mask, token_type_ids)
            loss = total_loss(INT_logit, INT_target, NER_logit, NER_target)

            epoch_loss += loss.item()
            total_len += 1
    return epoch_loss / total_len


if __name__ == '__main__':
    batch = pickle.load(open('small_batch_train.pkl', 'rb'))
    model = NLU_model()
    ids, mask, token_type_ids = batch['ids'], batch['mask'], batch['token_type_ids']
    INT_logit, NER_logit = model(ids, mask, token_type_ids)
    INT_target,NER_target= batch['INT_target'], batch['NER_target']
    print(total_loss(INT_logit, INT_target, NER_logit, NER_target))

    print('end')








