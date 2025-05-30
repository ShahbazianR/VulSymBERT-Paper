import os
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

#####GPU SELECTION######
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
if torch.cuda.is_available():
    device = torch.device('cuda') 
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using CPU')
print(f'Using device: {device}')


#####################################
# 0. GLOBAL CONFIG
#####################################
MODEL_NAME   = 'efederici/sentence-bert-base'
EPOCHS       = 50
BATCH_SIZE   = 32
MAX_LEN      = 256
LR           = 2e-5

DATASET_PATHS =[
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
            "./SBerpretmodels/AV/checkpoint_dataset1",
            "./SBertpremodels/AV/checkpoint_dataset2",
            "./SBertpremodels/AV/checkpoint_dataset3",
            "./SBertpremodels/AV/checkpoint_dataset4",
            "./SBertpremodels/AV/checkpoint_dataset5",
        ],
    },
    {
        'name':       'AC',
        'label_col':  'ac',
        'mapping':    {'L':0, 'H':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./SBertpremodels/AC/checkpoint_dataset1",
            "./SBertpremodels/AC/checkpoint_dataset2",
            "./SBertpremodels/AC/checkpoint_dataset3",
            "./SBertpremodels/AC/checkpoint_dataset4",
            "./SBertpremodels/AC/checkpoint_dataset5",
        ],
    },
    {
        'name':       'PR',
        'label_col':  'pr',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./SBertpremodels/PR/checkpoint_dataset1",
            "./SBertpremodels/PR/checkpoint_dataset2",
            "./SBertpremodels/PR/checkpoint_dataset3",
            "./SBertpremodels/PR/checkpoint_dataset4",
            "./SBertpremodels/PR/checkpoint_dataset5",
        ],
    },
    {
        'name':       'UI',
        'label_col':  'ui',
        'mapping':    {'N':0, 'R':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./SBertpremodels/UI/checkpoint_dataset1",
            "./SBertpremodels/UI/checkpoint_dataset2",
            "./SBertpremodels/UI/checkpoint_dataset3",
            "./SBertpremodels/UI/checkpoint_dataset4",
            "./SBertpremodels/UI/checkpoint_dataset5",
        ],
    },
    {
        'name':       'S',
        'label_col':  'S',
        'mapping':    {'U':0, 'C':1},
        'num_labels': 2,
        'ckpt_dirs': [
            "./SBertpremodels/S/checkpoint_dataset1",
            "./SBertpremodels/S/checkpoint_dataset2",
            "./SBertpremodels/S/checkpoint_dataset3",
            "./SBertpremodels/S/checkpoint_dataset4",
            "./SBertpremodels/S/checkpoint_dataset5",
        ],
    },
    {
        'name':       'I',
        'label_col':  'i',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./SBertpremodels/I/checkpoint_dataset1",
            "./SBertpremodels/I/checkpoint_dataset2",
            "./SBertpremodels/I/checkpoint_dataset3",
            "./SBertpremodels/I/checkpoint_dataset4",
            "./SBertpremodels/I/checkpoint_dataset5",
        ],
    },
    {
        'name':       'C',
        'label_col':  'c',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./SBertpremodels/C/checkpoint_dataset1",
            "./SBertpremodels/C/checkpoint_dataset2",
            "./SBertpremodels/C/checkpoint_dataset3",
            "./SBertpremodels/C/checkpoint_dataset4",
            "./SBertpremodels/C/checkpoint_dataset5",
        ],
    },
    {
        'name':       'A',
        'label_col':  'a',
        'mapping':    {'L':0, 'H':1, 'N':2},
        'num_labels': 3,
        'ckpt_dirs': [
            "./SBertpremodels/A/checkpoint_dataset1",
            "./SBertpremodels/A/checkpoint_dataset2",
            "./SBertpremodels/A/checkpoint_dataset3",
            "./SBertpremodels/A/checkpoint_dataset4",
            "./SBertpremodels/A/checkpoint_dataset5",
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
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

#####################################
# 2. MODEL DEFINITION WITH save_pretrained
#####################################
class SBertForClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        hidden_size     = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out     = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # save transformer backbone + config
        self.backbone.save_pretrained(save_directory)
        # save classifier head
        torch.save(self.classifier.state_dict(),
                   os.path.join(save_directory, "classifier.pt"))

    @classmethod
    def from_pretrained(cls, load_directory, model_name, num_labels):
        # init and load backbone
        model = cls(model_name, num_labels)
        model.backbone = AutoModel.from_pretrained(load_directory)
        # load classifier head
        classifier_path = os.path.join(load_directory, "classifier.pt")
        model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        return model

#####################################
# 3. TRAIN & LOG FUNCTION
#####################################
def train_and_log(model, train_loader, val_loader, test_loader,
                  device, phase_name, ds_idx, ckpt_dir,
                  epochs=EPOCHS, lr=LR):
    # freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    # optimizer only for classifier
    optimizer = AdamW(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(ckpt_dir, f"epoch_metrics_{phase_name}_ds{ds_idx}.csv")
    rows = []

    model.to(device)
    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        t_loss = t_corr = t_samp = 0
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items()}
            optimizer.zero_grad()
            logits = model(b['input_ids'], b['attention_mask'])
            loss   = criterion(logits, b['labels'])
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            preds   = logits.argmax(dim=1)
            t_corr += (preds == b['labels']).sum().item()
            t_samp += b['labels'].size(0)
        train_loss = t_loss / len(train_loader)
        train_acc  = t_corr / t_samp

        # VALIDATION
        model.eval()
        v_loss = v_corr = v_samp = 0
        with torch.no_grad():
            for b in val_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b['input_ids'], b['attention_mask'])
                loss   = criterion(logits, b['labels'])
                v_loss += loss.item()
                preds   = logits.argmax(dim=1)
                v_corr += (preds == b['labels']).sum().item()
                v_samp += b['labels'].size(0)
        val_loss = v_loss / len(val_loader)
        val_acc  = v_corr / v_samp if v_samp>0 else 0

        # TEST
        te_loss = te_corr = te_samp = 0
        all_p, all_l = [], []
        with torch.no_grad():
            for b in test_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b['input_ids'], b['attention_mask'])
                loss   = criterion(logits, b['labels'])
                te_loss += loss.item()
                preds   = logits.argmax(dim=1)
                te_corr += (preds == b['labels']).sum().item()
                te_samp += b['labels'].size(0)
                all_p.extend(preds.cpu().tolist())
                all_l.extend(b['labels'].cpu().tolist())
        test_loss = te_loss / len(test_loader)
        test_acc  = te_corr / te_samp if te_samp>0 else 0
        precision = precision_score(all_l, all_p, average='macro', zero_division=1)
        recall    = recall_score(all_l, all_p, average='macro')
        f1        = f1_score(all_l, all_p, average='macro')
        kappa     = cohen_kappa_score(all_l, all_p)

        print(f"[{phase_name}][DS{ds_idx}][Epoch {epoch}/{epochs}] "
              f"Tr {train_loss:.4f}/{train_acc:.4f} | "
              f"Va {val_loss:.4f}/{val_acc:.4f} | "
              f"Te {test_loss:.4f}/{test_acc:.4f}")

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
            'test_precision': precision,
            'test_recall':    recall,
            'test_f1':        f1,
            'test_kappa':     kappa,
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    return model

#####################################
# 4. MAIN LOOP
#####################################
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for phase in PHASES:
        phase_name = phase['name']
        prev_ckpt  = None

        for idx, csv_path in enumerate(DATASET_PATHS, start=1):
            ckpt_dir = phase['ckpt_dirs'][idx-1]
            print(f"\n=== Phase {phase_name} DS{idx}: {csv_path} ===")

            # load + preprocess
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

            # init or resume
            if prev_ckpt is None:
                model = SBertForClassification(MODEL_NAME, phase['num_labels'])
            else:
                model = SBertForClassification.from_pretrained(
                    prev_ckpt, MODEL_NAME, phase['num_labels']
                )

            # train & log
            model = train_and_log(
                model, train_ld, val_ld, test_ld,
                device, phase_name, idx, ckpt_dir
            )

            # ذخیرهٔ مدل و توکنایزر
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            prev_ckpt = ckpt_dir