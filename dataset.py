import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import csv
import random

# tokenizer
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

# data
data_path = './korean_smile_style_dataset/smile.csv'
f = open(data_path, 'r')
rdr = csv.reader(f)

session_dataset = []
session = []

for i, line in enumerate(rdr):
    if i == 0:
        header = line
    else:
        utt = line[0]
        if utt.strip() != '':
            session.append(utt)
        else:
            session_dataset.append(session)
            session = []
session_dataset.append(session)

all_utts = set()
for session in session_dataset:
    for utt in session:
        all_utts.add(utt)
f.close()

# 모듈
def get_mask(session):
    mask_ratio = 0.15
    corrupt_tokens = []
    output_tokens = []

    for i, utt in enumerate(session):
        original_token = tokenizer.encode(utt, add_special_tokens=False)

        mask_num = int(len(original_token) * mask_ratio)
        mask_positions = random.sample([x for x in range(len(original_token))], mask_num)
        corrupt_token = []
        for pos in range(len(original_token)):
            if pos in mask_positions:
                corrupt_token.append(tokenizer.mask_token_id)
            else:
                corrupt_token.append(original_token[pos])
        
        if i == len(session)-1:
            output_tokens += original_token
            corrupt_tokens += corrupt_token
        else:
            output_tokens += original_token + [tokenizer.sep_token_id]
            corrupt_tokens += corrupt_token + [tokenizer.sep_token_id]
    
    return output_tokens, corrupt_tokens


def get_urc(session):
    urc_tokens = []
    context_utts = []

    for i in range(len(session)):
        utt = session[i]
        original_token = tokenizer.encode(utt, add_special_tokens=False)

        if i == len(session)-1:
            # eos token
            urc_tokens += [tokenizer.eos_token_id]

            # positive(기존) response
            positive_tokens = [tokenizer.cls_token_id] + urc_tokens + original_token

            # random negative response
            while True:
                random_neg_response = random.choice(all_utts)
                if random_neg_response not in context_utts:
                    break
            random_neg_response_token = tokenizer.encode(random_neg_response, add_special_tokens=False)
            random_tokens = [tokenizer.cls_token_id] + urc_tokens + random_neg_response_token

            # context negative response
            context_neg_response = random.choice(context_utts)
            context_neg_response_token = tokenizer.encode(context_neg_response, add_special_tokens=False)
            context_neg_tokens = [tokenizer.cls_token_id] + urc_tokens + context_neg_response_token
        else:
            urc_tokens += original_token + [tokenizer.sep_token_id]
        
        context_utts.append(utt)
    
    return positive_tokens, random_tokens, context_neg_tokens

# Build dataset
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        special_tokens = {'sep_token':'<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)

        # data read
        f = open(data_path, 'r')
        rdr = csv.reader(f)

        # session별 dataset 만들기
        session = []
        session_dataset = []

        for idx, line in enumerate(rdr):
            if idx == 0:
                header = line
            else:
                utt = line[0] # 첫번째 발화형식 저장
                if utt.strip() != '':
                    session.append(utt)
                else:
                    session_dataset.append(session)
                    session = []
        session_dataset.append(session)
        f.close()

        # short session context : 길이는 4(context=3, response=1)
        k = 4
        self.short_session_dataset = []
        for session in session_dataset:
            for i in range(len(session)-k+1):
                self.short_session_dataset.append(session[i:i+k])
        
        # all utt for random negative response
        self.all_utts = set()
        for session in session_dataset:
            for utt in session:
                self.all_utts.add(utt)
        self.all_utts = list(self.all_utts)
    

    def __len__(self):
        return len(self.short_session_dataset)
    

    def __getitem__(self, idx):
        session = self.short_session_dataset[idx]

        # MLM
        self.output_tokens, self.corrupt_tokens = get_mask(session)

        # label for loss
        self.corrupt_mask_positions = []
        for pos in range(len(self.corrupt_tokens)):
            if self.corrupt_tokens[pos] == self.tokenizer.mask_token_id:
                self.corrupt_mask_positions.append(pos)
        
        # URC
        self.positive_tokens, self.random_tokens, self.context_neg_tokens = get_urc(session)

        return self.corrupt_tokens, self.output_tokens, self.corrupt_mask_positions, [self.positive_tokens, self.random_tokens, self.context_neg_tokens], [0, 1, 2]
    

    def collate_fn(self, sessions):
        '''
            input:
                data: [(session1), (session2), ... ]
            return:
                batch_corrupt_tokens: (B, L) padded
                batch_output_tokens: (B, L) padded
                batch_corrupt_mask_positions: list
                batch_urc_inputs: (B, L) padded
                batch_urc_labels: (B)
                batch_mlm_attentions
                batch_urc_attentions

            만약 batch가 3이라면,
            MLM = 3개의 입력데이터 (입력데이터별로 길이가 다름)
            URC = 9개의 입력데이터 (context는 길이가 다름, response candidate도 길이가 다름)
        '''

        batch_corrupt_tokens, batch_output_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_urc_labels = [], [], [], [], [] # dataset output
        batch_mlm_attentions, batch_urc_attentions = [], []

        # max length 찾기 -> for padding       
        corrupt_max_len = 0
        urc_max_len = 0
        for session in sessions:
            corrupt_tokens, output_tokens, corrupt_mask_positions, urc_inputs, urc_labels = session # dataset output
            if len(corrupt_tokens) > corrupt_max_len:
                corrupt_max_len = len(corrupt_tokens)
            positive_tokens, random_tokens, context_neg_tokens = urc_inputs
            if max(len(positive_tokens), len(random_tokens), len(context_neg_tokens)) > urc_max_len:
                urc_max_len = max(len(positive_tokens), len(random_tokens), len(context_neg_tokens))
        
        # padding tokens 추가
        for session in sessions:
            corrupt_tokens, output_tokens, corrupt_mask_positions, urc_inputs, urc_labels = session # dataset output

            # MLM 입력
            batch_corrupt_tokens.append(corrupt_tokens + [self.tokenizer.pad_token_id for _ in range(corrupt_max_len - len(corrupt_tokens))])
            batch_mlm_attentions.append([1 for _ in range(len(corrupt_tokens))] + [0 for _ in range(corrupt_max_len - len(corrupt_tokens))])
            # MLM 출력
            batch_output_tokens.append(output_tokens + [self.tokenizer.pad_token_id for _ in range(corrupt_max_len - len(corrupt_tokens))])
            # MLM label
            batch_corrupt_mask_positions.append(corrupt_mask_positions)
            # URC 입력
            for urc_input in urc_inputs:
                batch_urc_inputs.append(urc_input + [self.tokenizer.pad_token_id for _ in range(urc_max_len - len(urc_input))])
                batch_urc_attentions.append([1 for _ in range(len(urc_input))] + [0 for _ in range(urc_max_len - len(urc_input))])
            # URC label
            batch_urc_labels += urc_labels
        
        return torch.tensor(batch_corrupt_tokens), torch.tensor(batch_output_tokens), batch_corrupt_mask_positions, torch.tensor(batch_urc_inputs), torch.tensor(batch_urc_labels), torch.tensor(batch_mlm_attentions), torch.tensor(batch_urc_attentions)