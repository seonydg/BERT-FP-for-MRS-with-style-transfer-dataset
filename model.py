import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM, AutoTokenizer

class PostModel(nn.Module):
    def __init__(self):
        super(PostModel, self).__init__()

        self.model = RobertaForMaskedLM.from_pretrained('klue/roberta-base')
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        
        special_tokens = {'sep_token':'<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.hiddenDim = self.model.config.hidden_size
        self.W = nn.Linear(self.hiddenDim, 3) # score matrix
    
    def forward(self, batch_corrupt_tokens, batch_corrupt_mask_positions, batch_urc_inputs, batch_mlm_attentions, batch_urc_attentions):
        # MLM
        corrupt_outputs = self.model(batch_corrupt_tokens, attention_mask=batch_mlm_attentions)['logits'] # [B, L, vocab_size(32001)]
        corrupt_mask_outputs = []
        for idx in range(len(batch_corrupt_mask_positions)):
            corrupt_mask_output = []
            batch_corrupt_mask_position = batch_corrupt_mask_positions[idx]
            for pos in batch_corrupt_mask_position:
                corrupt_mask_output.append(corrupt_outputs[idx, pos, :].unsqueeze(0)) # [1, 32001]
            corrupt_mask_outputs.append(torch.cat(corrupt_mask_output, 0)) # [mask_num, 32001]
        
        # URC
        urc_outputs = self.model(batch_urc_inputs, attention_mask=batch_urc_attentions, output_hidden_states=True)['hidden_states'][-1] # batch = 2 -> [6, 61, 768]
        urc_logits = self.W(urc_outputs) # [6, 61, 3]
        urc_cls_outputs = urc_logits[:, 0, :] # cls tokens : [6, 3]

        return corrupt_mask_outputs, urc_cls_outputs


from transformers import RobertaForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn

class FT_Model(nn.Module):
    def __init__(self):
        super(FT_Model, self).__init__()

        self.model = RobertaForMaskedLM.from_pretrained('klue/roberta-base')
        self.hiddenDim = self.model.config.hidden_size
        
        self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
        special_tokens = {'sep_token':'<SEP>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_embeddings(len(self.tokenizer))

        # score matrix
        self.W2 = nn.Linear(self.hiddenDim, 2)
    
    def forward(self, batch_input_tokens, batch_input_attentions):
        outputs = self.model(batch_input_tokens, attention_mask=batch_input_attentions, output_hidden_states=True)['hidden_states'][-1] # (B, L, hidden_dim)
        cls_outputs = outputs[:, 0, :] # (B, hidden_dim)
        cls_logits = self.W2(cls_outputs) # (B, 2(class))

        return cls_logits