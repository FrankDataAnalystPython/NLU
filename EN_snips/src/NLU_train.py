import config
import dataset
import engine
import torch
from model import *
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import time
import os
import numpy as np
import pickle

with open(config.TRAIN_FILE_X, 'r') as f:
    train_in = f.readlines()
with open(config.TRAIN_FILE_INT, 'r') as f:
    train_int = f.readlines()
with open(config.TRAIN_FILE_NER, 'r') as f:
    train_ner = f.readlines()

with open(config.VALID_FILE_X, 'r') as f:
    valid_in = f.readlines()
with open(config.VALID_FILE_INT, 'r') as f:
    valid_int = f.readlines()
with open(config.VALID_FILE_NER, 'r') as f:
    valid_ner = f.readlines()

INT_encoder = config.INT_encoder
NER_encoder = config.NER_encoder

train_data = dataset.NLU_dataset(train_in, train_int, train_ner)
valid_data = dataset.NLU_dataset(valid_in, valid_int, valid_ner)

train_iterator = DataLoader(train_data,
                            sampler = RandomSampler(train_data),
                            batch_size = config.BATCH_SIZE
                            )

valid_iterator = DataLoader(valid_data,
                            sampler = RandomSampler(valid_data),
                            batch_size = config.BATCH_SIZE
                            )

batch = next(iter(valid_iterator))
pickle.dump(batch, open('small_batch_valid.pkl', 'wb'))
batch = next(iter(train_iterator))
pickle.dump(batch, open('small_batch_train.pkl', 'wb'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NLU_model()
if config.use_pretrained:
    model.load_state_dict(torch.load(config.MODEL_PATH,
                                     map_location = torch.device(device)
                                     ))
optimizer = AdamW(model.parameters(), lr = config.LR)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 5,
                                            num_training_steps = len(train_iterator) * config.N_EPOCHS
                                            )
n_epoch_stop = config.N_EPOCHS_STOP
min_loss = np.Inf

for epoch in range(config.N_EPOCHS):
    start_time = time.time()
    train_loss = engine.train(
        model,
        train_iterator,
        optimizer,
        device,
        scheduler,
        (epoch + 1, config.N_EPOCHS)
    )
    torch.save(model.state_dict(), config.MODEL_PATH)
    valid_loss = engine.evaluate(
        model,
        valid_iterator,
        device
    )
    end_time = time.time()
    epoch_mins, epoch_secs = engine.epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
          # f'| Train Bleu: {train_score * 100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f}')
          # f'| Valid Bleu: {valid_score * 100:.2f}%')

    if valid_loss < min_loss:
        epochs_no_improve = 0
        min_val_loss = valid_loss
        torch.save(model.state_dict(), 'NLU_earlier.model')
    else:
        epochs_no_improve += 1

    if epoch > 10 and epochs_no_improve == n_epoch_stop:
        print('Early Stopping')
        break

print('end')


