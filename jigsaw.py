import numpy as np
import pandas as pd

import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F 

from sklearn import metrics
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from apex import amp
import random 
import argparse

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

seed = 1110
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    return args

device=torch.device('cuda')
max_seq_len = 220
data_dir = '/home/jyzhao/git8/data/jigsawUnintendedBiasinToxicityClassification'
output_dir = '/home/jyzhao/git8/disentanglement'

TOXICITY_COLUMN = 'target'

bert_model = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

#load csv file
num_to_load = 100000

train_df_raw = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir,"test_public_expanded.csv"))
    #convert comment_text to strings in df

train_df_raw['comment_text'] = train_df_raw['comment_text'].astype(str)
test_df['comment_text'] = test_df['comment_text'].astype(str)

# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df_raw=train_df_raw.fillna(0)
# convert target to 0,1
train_df_raw['target']=(train_df_raw['target']>=0.5).astype(float)

test_df = test_df.fillna(0)
test_df['target']=(test_df['toxicity']>=0.5).astype(float)

# input_ids = train_df['comment_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens = True, max_length = max_seq_len, pad_to_max_length = True))
# input_ids = np.array([i for i in input_ids.values])
# np.save('/home/jyzhao/git8/disentanglement/padded', input_ids)
input_ids_raw = np.load('/local/jyzhao/Github/disentanglement/padded.npy')
raw_labels = train_df_raw['target'].values.flatten()
pos_raw = np.where(raw_labels == 1.0)[0]
neg_raw = np.where(raw_labels == 0.0)[0]
white_idx = train_df_raw[train_df_raw['white'] >= 0.5].index.values
black_idx = train_df_raw[train_df_raw['black'] >= 0.5].index.values
nidx = np.random.choice(neg_raw, len(pos_raw))
pidx = np.random.choice(pos_raw, len(pos_raw)) ###to balance the dataset!!
idxs = np.unique(np.concatenate((nidx, pidx, white_idx, black_idx), axis = 0))

# np.random.shuffle(idxs)
train_df = train_df_raw.iloc[idxs]
input_ids = input_ids_raw[idxs]
assert len(input_ids) == len(train_df)
print(f"{len(train_df[train_df['target'] == 1.0 ])} pos; {len(train_df[train_df['target'] == 0.0])} neg")

test_input_ids = np.load('/local/jyzhao/Github/disentanglement/test_padded.npy')
print('loaded %d train & %d test records' % (len(train_df), len(test_input_ids)))

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, train_df['target'].values.flatten(), random_state=56, test_size=0.2)

train_data = Data.TensorDataset(torch.tensor(train_inputs, dtype = torch.long), torch.tensor(train_labels, dtype = torch.long))
val_data = Data.TensorDataset(torch.tensor(validation_inputs, dtype = torch.long), torch.tensor(validation_labels, dtype = torch.long)) 
test_data = Data.TensorDataset(torch.tensor(test_input_ids, dtype = torch.long), torch.tensor(test_df['target'].values.flatten(), dtype = torch.long)) 

epochs = 2
lr=2e-5
batch_size = 20
adam_epsilon = 1e-8
torch.backends.cudnn.deterministic = True

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels = 2, output_hidden_states=True)
model.zero_grad()
model = model.to(device)

def epoch_pass(epoch, data_loader, model, optimizer, scheduler, istrain=False, lossf = None):
    if istrain:
        model.train()
    else:
        model.eval()
    avg_loss = 0.
    correct = 0.0
    total = 0
    preds = []
    hiddens = []
    label_scores = []
    tk0 = tqdm(enumerate(data_loader),total=len(data_loader),leave=True)
    for i,(x_batch, y_batch) in tk0:
        outputs = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=y_batch.to(device))
        loss = outputs[0]
        prediction = torch.argmax(outputs[1],dim=1)
        pred_score = outputs[1][:, 1]
        if istrain:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()                         
            optimizer.zero_grad()
            
        if lossf:
            lossf = 0.98*lossf+0.02*loss.item()
        else:
            lossf = loss.item()
        tk0.set_postfix(loss = lossf)
        avg_loss += loss.item() 
        correct += (prediction==y_batch.to(device)).sum().item()
        total += y_batch.size(0)

        preds.extend(prediction.cpu().detach().numpy())
        label_scores.extend(pred_score.cpu().detach().numpy())
        bert_mask = torch.FloatTensor((x_batch>0).float()).unsqueeze(2)
        second2last = outputs[2][-2]
        nopadded = torch.sum(torch.mul(second2last,bert_mask.cuda()), axis = 1) / bert_mask.cuda().sum(1).view(len(x_batch), -1)
        hiddens.extend(nopadded.cpu().detach().numpy())
        # hiddens.extend(outputs[2][-1][:,0,:].cpu().detach().numpy())
    avg_accuracy = correct / total
    avg_loss /= total
    return avg_loss, avg_accuracy, preds, hiddens, label_scores


def main():
    args = get_args()
    if args.train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)

        total_steps = len(train_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        tq = tqdm(range(epochs))
        for epoch in tq:
            model.train()
            best_acc = 0.0
            best_model_epoch = -1
            
            train_loss, train_acc, _, _, _= epoch_pass(epoch, train_loader, model, optimizer, scheduler,  True)    
            with torch.no_grad():
                dev_loss, dev_acc, _, _, _ = epoch_pass(0, val_loader, model, None, None, False)
            tq.set_postfix(train_loss=train_loss,train_accuracy=train_acc, dev_loss = dev_loss, dev_acc = dev_acc)
            if dev_acc > best_acc:
                best_acc = dev_acc
                best_model_epoch = epoch
                torch.save({'epoch':epoch, 'state_dict':model.state_dict()}, output_dir+'/bert_withBlackandWhite.pth.tar')
            print(f"dev loss{dev_loss}, dev acc:{dev_acc}")
    print("Testing model")
    model.load_state_dict(torch.load('/local/jyzhao/Github/disentanglement/bert_withBlackandWhite.pth.tar')['state_dict'])
    with torch.no_grad():
        test_loss, test_acc, test_preds, test_embs, test_scores = epoch_pass(0, test_loader, model, None, False)
    print(f"test loss:{test_loss}, acc:{test_acc}")
    test_df['preds'] = test_preds
    test_df["pred_score"] = test_scores
    test_df.to_csv('pred_test_bw_withpredscores.csv')
    # np.save('./test_emb_bw_2nd2last', test_embs)

if __name__ == '__main__':
    main()