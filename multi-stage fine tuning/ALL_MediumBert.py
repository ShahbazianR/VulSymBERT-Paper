import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

#####GPU SELECTION######
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if torch.cuda.is_available():
    device = torch.device('cuda') 
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using CPU')
print(f'Using device: {device}')



# Paths to your datasets
DATASET_PATHS = [
    "./Dataset/4.csv",
    "./Dataset/3.csv",
    "./Dataset/2.csv",
    "./Dataset/1.csv",
    "./Dataset/5.csv",
]
# per-phase configuration
PHASES = [
    {
        'name':       'AV',
        'label_col':  'av',
        'mapping':    {'N':0, 'L':1, 'A':2, 'P':3},
        'num_labels': 4,
        'ckpt_base':  './Mediumbertmodels/AV'
    },
    {
        'name':       'AC',
        'label_col':  'ac',
        'mapping':    {'L':0, 'H':1},
        'num_labels': 2,
        'ckpt_base':  './Mediumbertmodels/AC'
    },
    {
        'name':       'PR',
        'label_col':  'pr',
        'mapping':    {'N':0, 'L':1, 'H':2},
        'num_labels': 3,
        'ckpt_base':  './Mediumbertmodels/PR'
    },
    {
        'name':       'UI',
        'label_col':  'ui',
        'mapping':    {'N':0, 'R':1},
        'num_labels': 2,
        'ckpt_base':  './Mediumbertmodels/UI'
    },
    {
        'name':       'S',
        'label_col':  's',
        'mapping':    {'U':0, 'C':1},
        'num_labels': 2,
        'ckpt_base':  './Mediumbertmodels/S'
    },
    {
        'name':       'C',
        'label_col':  'c',
        'mapping':    {'N':0, 'L':1, 'H':2},
        'num_labels': 3,
        'ckpt_base':  './Mediumbertmodels/C'
    },
    {
        'name':       'I',
        'label_col':  'i',
        'mapping':    {'N':0, 'L':1, 'H':2},
        'num_labels': 3,
        'ckpt_base':  './Mediumbertmodels/I'
    },
    {
        'name':       'A',
        'label_col':  'a',
        'mapping':    {'N':0, 'L':1, 'H':2},
        'num_labels': 3,
        'ckpt_base':  './Mediumbertmodels/A'
    },
]

# Hyperparameters
MODEL_NAME = 'prajjwal1/bert-medium'
EPOCHS     = 50
BATCH_SIZE = 32
MAX_LEN    = 256
LR         = 2e-5

# Dataset class
class VulDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(label, dtype=torch.long)
        }

# Training & logging function
def train_and_log(model, train_loader, val_loader, test_loader,
                  device, phase_name, ds_idx, ckpt_dir,
                  epochs=EPOCHS, lr=LR):
    optimizer = AdamW(model.parameters(), lr=lr)
    os.makedirs(ckpt_dir, exist_ok=True)
    rows = []

    for epoch in range(1, epochs+1):
        # — TRAIN —
        model.train()
        total_loss = correct = samples = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss, logits = out.loss, out.logits
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch['labels']).sum().item()
            samples += len(batch['labels'])
        train_loss = total_loss / len(train_loader)
        train_acc  = correct / samples

        # — VALIDATION —
        model.eval()
        val_loss = val_corr = val_samp = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                l, logits = out.loss, out.logits
                val_loss += l.item()
                preds = torch.argmax(logits, dim=1)
                val_corr += (preds == batch['labels']).sum().item()
                val_samp += len(batch['labels'])
        val_loss = val_loss / len(val_loader)
        val_acc  = val_corr / val_samp if val_samp>0 else 0

        # — TEST —
        test_loss = test_corr = test_samp = 0
        all_preds = []; all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                l, logits = out.loss, out.logits
                test_loss += l.item()
                preds = torch.argmax(logits, dim=1)
                test_corr += (preds == batch['labels']).sum().item()
                test_samp += len(batch['labels'])
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        test_loss  = test_loss / len(test_loader)
        test_acc   = test_corr / test_samp if test_samp>0 else 0
        test_prec  = precision_score(all_labels, all_preds, average='macro', zero_division=1)
        test_rec   = recall_score(all_labels, all_preds, average='macro')
        test_f1    = f1_score(all_labels, all_preds, average='macro')
        test_kappa = cohen_kappa_score(all_labels, all_preds)

        # console log
        print(f"[{phase_name}][DS{ds_idx}][Epoch {epoch}/{epochs}] "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.4f} | "
              f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

        # collect metrics
        rows.append({
            'phase':          phase_name,
            'dataset_idx':    ds_idx,
            'epoch':          epoch,
            'train_loss':     train_loss,
            'train_acc':      train_acc,
            'val_loss':       val_loss,
            'val_acc':        val_acc,
            'test_loss':      test_loss,
            'test_acc':       test_acc,
            'test_precision': test_prec,
            'test_recall':    test_rec,
            'test_f1':        test_f1,
            'test_kappa':     test_kappa,
        })

    # write per-dataset CSV
    df = pd.DataFrame(rows)
    ds_csv = os.path.join(ckpt_dir, f'epoch_metrics_{phase_name}_dataset{ds_idx}.csv')
    df.to_csv(ds_csv, index=False)
    print(f"Saved per-dataset CSV: {ds_csv}")

    return model, rows

# Entry point
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    for phase in PHASES:
        phase_rows = []
        previous_ckpt = None

        for idx, csv_path in enumerate(DATASET_PATHS):
            ds_idx   = idx + 1
            ckpt_dir = os.path.join(phase['ckpt_base'], f'checkpoint_dataset{ds_idx}')

            # load & map labels
            df = pd.read_csv(csv_path)
            df[phase['label_col']] = df[phase['label_col']].map(phase['mapping'])
            if df[phase['label_col']].isnull().any():
                raise ValueError(f"Unmapped labels for phase {phase['name']} in {csv_path}")

            texts  = df['text'].astype(str).values
            labels = df[phase['label_col']].values

            # prepare data loaders
            ds = VulDataset(texts, labels, tokenizer)
            n  = len(ds)
            t, v, te = int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)
            train_ds, val_ds, test_ds = random_split(
                ds, [t, v, te], generator=torch.Generator().manual_seed(42)
            )
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

            # load or resume model
            if idx == 0:
                model = BertForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=phase['num_labels']
                )
            else:
                model = BertForSequenceClassification.from_pretrained(
                    previous_ckpt, num_labels=phase['num_labels']
                )
            model.to(device)

            # train & log
            model, rows = train_and_log(
                model, train_loader, val_loader, test_loader,
                device, phase['name'], ds_idx, ckpt_dir
            )
            phase_rows.extend(rows)

            # save for next dataset
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            previous_ckpt = ckpt_dir

        # write aggregated CSV for this phase
        phase_csv = os.path.join(
            phase['ckpt_base'],
            f'all_epoch_metrics_{phase["name"]}.csv'
        )
        pd.DataFrame(phase_rows).to_csv(phase_csv, index=False)
        print(f"Saved aggregated CSV for phase {phase['name']}: {phase_csv}")
