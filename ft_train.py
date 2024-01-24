import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from transformers import RobertaForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

from model import FTDataset
from model import PostModel, FT_Model

import json
import os
import random
from tqdm import tqdm



def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val


def ft_SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'ft_model.bin'))


# R@1 eval 사용
def CalR1(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for idx, data in enumerate(tqdm(dataloader, desc='evaluation')):
            batch_input_tokens, batch_input_attentions, batch_input_labels = data

            batch_input_tokens = batch_input_tokens.cuda()
            batch_input_attentions = batch_input_attentions.cuda()
            batch_input_labels = batch_input_labels.cuda()

            # pred
            outputs = model(batch_input_tokens, batch_input_attentions)
            probs = softmax(outputs, 1)
            true_probs = probs[:, 1] # positive 확률
            pred_idx =true_probs.argmax(0).item()
            gt_idx = batch_input_labels.argmax(0).item()

            if pred_idx == gt_idx:
                correct += 1
    
    return round(correct/len(dataloader)*100, 2)

# dataloader
# train
train_path = '/content/korean_smile_style_dataset/train.json'
train_dataset = FTDataset(train_path)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)

# valid
valid_path = '/content/korean_smile_style_dataset/valid.json'
valid_dataset = FTDataset(valid_path)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=valid_dataset.collate_fn)

# modeling
ft_model = FT_Model().cuda()
ft_model.load_state_dict(torch.load('post_model.bin'), strict=False)

# 하이퍼 파라미터
training_epochs = 10
max_grad_norm = 10
lr = 1e-5
num_training_steps = len(train_dataset) * training_epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

best_score = -1

for epoch in range(training_epochs):

    ft_model.train()
    for idx, data in enumerate(tqdm(train_dataloader)):
        batch_input_tokens, batch_input_attentions, batch_input_labels = data

        batch_input_tokens = batch_input_tokens.cuda()
        batch_input_attentions = batch_input_attentions.cuda()
        batch_input_labels = batch_input_labels.cuda()

        #pred
        outputs = ft_model(batch_input_tokens, batch_input_attentions)
        loss_val = CELoss(outputs, batch_input_labels)

        optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
    
    ft_model.eval()
    r1 = CalR1(ft_model, valid_dataloader)
    print(f'Epoch : {epoch+1} | R@1 score : {r1}')

    if r1 > best_score:
        best_score = r1
        print(f'Best score : {r1}')

        ft_SaveModel(ft_model, '.')