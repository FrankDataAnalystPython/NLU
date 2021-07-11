import os
import tokenizers
from sklearn.preprocessing import LabelEncoder

MAX_LEN = 64
BATCH_SIZE = 32
N_EPOCHS = 10
LR = 3e-5
BASE_MODEL_PATH = '../input/en_bert_base'
MODEL_PATH = 'NLU.model'
TRAIN_FILE_X = '../input/train/seq.in'
TRAIN_FILE_INT = '../input/train/label'
TRAIN_FILE_NER = '../input/train/seq.out'
VALID_FILE_X = '../input/test/seq.in'
VALID_FILE_INT = '../input/test/label'
VALID_FILE_NER = '../input/test/seq.out'
PICKLE_FILE = 'train_valid.pkl'
N_EPOCHS_STOP = 10
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BASE_MODEL_PATH, 'vocab.txt'),
    lowercase = True
)
use_pretrained = False


with open('../input/intent_label.txt', 'r') as f:
    INT_label = f.readlines()
INT_label = [i.replace('\n', '') for i in INT_label]
with open('../input/slot_label.txt', 'r') as f:
    NER_label = f.readlines()
NER_label = [i.replace('\n', '') for i in NER_label]
INT_encoder = LabelEncoder().fit(INT_label)
NER_encoder = LabelEncoder().fit(NER_label)
NER_PAD_IDX = NER_encoder.transform(['PAD'])[0]

