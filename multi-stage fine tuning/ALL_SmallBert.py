import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

##### GPU SELECTION #####
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f" GPU: {torch.cuda.get_device_name(0)}")

#####################################
# 0. GLOBAL CONFIG
#####################################
MODEL_NAME = 'prajjwal1/bert-small'
EPOCHS     = 50
BATCH_SIZE = 32
MAX_LEN    = 256
LR         = 2e-5

DATASET_PATHS = [
    "./Dataset/4.csv",
    "./Dataset/3.csv",
    "./Dataset/2.csv",
    "./Dataset/1.csv",
    "./Dataset/5.csv",
]

PHASES = [
    {
        'name':       'AV',
        'label_col':  'av',
        'mapping':    {'N':0, 'L':1, 'A':2, 'P':3},
        'num_labels': 4,
        'ckpt_dirs': [
            "./Smallbertmodels/AV/checkpoint_dataset1",
            "./Smallbertmodels/AV/checkpoint_dataset2",
            "./Smallbertmodels/AV/checkpoint_dataset3",
            "./Smallbertmodels/AV/checkpoint_dataset4",
            "./Smallbertmodels/AV/checkpoint_dataset5",
        ],
    },
    {
        'name':       'AC',
        'label_col':  'ac',
        'mapping':    {'L':0, 'H':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./Smallbertmodels/AC/checkpoint_dataset1",
            "./Smallbertmodels/AC/checkpoint_dataset2",
            "./Smallbertmodels/AC/checkpoint_dataset3",
            "./Smallbertmodels/AC/checkpoint_dataset4",
            "./Smallbertmodels/AC/checkpoint_dataset5",
        ],
    },
    {
        'name':       'PR',
        'label_col':  'pr',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./Smallbertmodels/PR/checkpoint_dataset1",
            "./Smallbertmodels/PR/checkpoint_dataset2",
            "./Smallbertmodels/PR/checkpoint_dataset3",
            "./Smallbertmodels/PR/checkpoint_dataset4",
            "./Smallbertmodels/PR/checkpoint_dataset5",
        ],
    },
    {
        'name':       'UI',
        'label_col':  'ui',
        'mapping':    {'N':0, 'R':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./Smallbertmodels/UI/checkpoint_dataset1",
            "./Smallbertmodels/UI/checkpoint_dataset2",
            "./Smallbertmodels/UI/checkpoint_dataset3",
            "./Smallbertmodels/UI/checkpoint_dataset4",
            "./Smallbertmodels/UI/checkpoint_dataset5",
        ],
    },
    {
        'name':       'S',
        'label_col':  's',
        'mapping':    {'U':0, 'C':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./Smallbertmodels/S/checkpoint_dataset1",
            "./Smallbertmodels/S/checkpoint_dataset2",
            "./Smallbertmodels/S/checkpoint_dataset3",
            "./Smallbertmodels/S/checkpoint_dataset4",
            "./Smallbertmodels/S/checkpoint_dataset5",
        ],
    },
    {
        'name':       'I',
        'label_col':  'i',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./Smallbertmodels/I/checkpoint_dataset1",
            "./Smallbertmodels/I/checkpoint_dataset2",
            "./Smallbertmodels/I/checkpoint_dataset3",
            "./Smallbertmodels/I/checkpoint_dataset4",
            "./Smallbertmodels/I/checkpoint_dataset5",
        ],
    },
    {
        'name':       'C',
        'label_col':  'c',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./Smallbertmodels/C/checkpoint_dataset1",
            "./Smallbertmodels/C/checkpoint_dataset2",
            "./Smallbertmodels/C/checkpoint_dataset3",
            "./Smallbertmodels/C/checkpoint_dataset4",
            "./Smallbertmodels/C/checkpoint_dataset5",
        ],
    },
    {
        'name':       'A',
        'label_col':  'a',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./Smallbertmodels/A/checkpoint_dataset1",
            "./Smallbertmodels/A/checkpoint_dataset2",
            "./Smallbertmodels/A/checkpoint_dataset3",
            "./Smallbertmodels/A/checkpoint_dataset4",
            "./Smallbertmodels/A/checkpoint_dataset5",
        ],
    },
]

#####################################
# 1. DATASET CLASS
#####################################
class VulDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens   = True,
            max_length           = self.max_len,
            padding              = 'max_length',
            truncation           = True,
            return_attention_mask= True,
            return_tensors       = 'pt',
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }

#####################################
# 2. TRAIN & LOG FUNCTION
#####################################
def train_and_log(model,
                  train_loader, val_loader, test_loader,
                  device,
                  phase_name, ds_idx, ckpt_dir,
                  epochs=EPOCHS, lr=LR):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, f"epoch_metrics_{phase_name}_ds{ds_idx}.csv")
    rows = []

    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        t_loss = 0; t_corr = 0; t_samp = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**{k: batch[k] for k in ['input_ids','attention_mask','labels']})
            loss, logits = out.loss, out.logits
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            preds  = logits.argmax(dim=1)
            t_corr += (preds == batch['labels']).sum().item()
            t_samp += batch['labels'].size(0)
        train_loss = t_loss / len(train_loader)
        train_acc  = t_corr / t_samp

        # VALIDATION
        model.eval()
        v_loss = 0; v_corr = 0; v_samp = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**{k: batch[k] for k in ['input_ids','attention_mask','labels']})
                l, logits = out.loss, out.logits
                v_loss += l.item()
                preds  = logits.argmax(dim=1)
                v_corr += (preds == batch['labels']).sum().item()
                v_samp += batch['labels'].size(0)
        val_loss = v_loss / len(val_loader)
        val_acc  = v_corr / v_samp if v_samp>0 else 0

        # TEST
        model.eval()
        te_loss = 0; te_corr = 0; te_samp = 0
        all_preds = []; all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**{k: batch[k] for k in ['input_ids','attention_mask','labels']})
                l, logits = out.loss, out.logits
                te_loss += l.item()
                preds   = logits.argmax(dim=1)
                te_corr += (preds == batch['labels']).sum().item()
                te_samp += batch['labels'].size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())
        test_loss = te_loss / len(test_loader)
        test_acc  = te_corr / te_samp if te_samp>0 else 0
        test_prec = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        test_rec  = recall_score(all_labels, all_preds, average='macro')
        test_f1   = f1_score(all_labels, all_preds, average='macro')
        test_kap  = cohen_kappa_score(all_labels, all_preds)

        print(f"[{phase_name}][DS{ds_idx}][Epoch {epoch}/{epochs}] "
              f"Tr {train_loss:.4f}/{train_acc:.4f} | "
              f"Va {val_loss:.4f}/{val_acc:.4f} | "
              f"Te {test_loss:.4f}/{test_acc:.4f}")

        rows.append({
            'phase':           phase_name,
            'dataset_index':   ds_idx,
            'epoch':           epoch,
            'train_loss':      train_loss,
            'train_acc':       train_acc,
            'val_loss':        val_loss,
            'val_acc':         val_acc,
            'test_loss':       test_loss,
            'test_acc':        test_acc,
            'test_precision':  test_prec,
            'test_recall':     test_rec,
            'test_f1':         test_f1,
            'test_kappa':      test_kap,
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    return model

#####################################
# 3. MAIN LOOP
#####################################
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    for phase in PHASES:
        phase_name    = phase['name']
        prev_ckpt_dir = None

        for idx, csv_path in enumerate(DATASET_PATHS, start=1):
            ckpt_dir = phase['ckpt_dirs'][idx-1]
            print(f"\n=== Phase {phase_name} DS{idx}: {csv_path} ===")

            # load & map labels
            df = pd.read_csv(csv_path)
            df[phase['label_col']] = df[phase['label_col']].map(phase['mapping'])
            texts  = df['text'].astype(str).tolist()
            labels = df[phase['label_col']].tolist()

            # split
            ds = VulDataset(texts, labels, tokenizer)
            n  = len(ds)
            t, v, te = int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)
            train_ds, val_ds, test_ds = random_split(
                ds, [t, v, te], generator=torch.Generator().manual_seed(42)
            )
            train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
            test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

            # init or load model
            if prev_ckpt_dir is None:
                model = BertForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=phase['num_labels']
                )
            else:
                model = BertForSequenceClassification.from_pretrained(
                    prev_ckpt_dir, num_labels=phase['num_labels']
                )
            model.to(device)

            # train & log
            model = train_and_log(
                model,
                train_ld, val_ld, test_ld,
                device,
                phase_name, idx, ckpt_dir,
                epochs=EPOCHS, lr=LR
            )

            # save for next phase
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            prev_ckpt_dir = ckpt_dir
