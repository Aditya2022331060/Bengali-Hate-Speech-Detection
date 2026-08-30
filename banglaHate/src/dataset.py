# === banglaHate/src/dataset.py ===
"""
BanglaHateDataset — Unified PyTorch Dataset for multi-task classification
and explanation generation.

Handles:
  - Bengali text normalization
  - Tokenization via BanglaBERT tokenizer
  - Multi-task label encoding (type, target, severity)
  - Optional silver-standard explanation tokenization for generative head
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import pandas as pd


class BanglaHateDataset(Dataset):
    """
    Unified dataset for the Consistency-Constrained MTL model.

    Loads data from either CSV or JSON format and provides:
      - Tokenized input text (input_ids, attention_mask)
      - Encoded labels for 3 classification tasks
      - Optional tokenized explanations for the generative head

    Label encoding order (5-class taxonomy):
      type_of_hate:     ['None', 'Abusive', 'Political Hate', 'Religious Hate', 'Gender Hate']
      target_of_hate:   ['None', 'Individual', 'Organization', 'Community', 'Society']
      severity_of_hate: ['Little to None', 'Mild', 'Severe']
    """

    # Canonical label orderings — these define the index positions
    TYPE_LABELS = ['None', 'Abusive', 'Political Hate', 'Religious Hate', 'Gender Hate']

    # Remap legacy labels from original BanglaMultiHate dataset
    _LABEL_REMAP = {'Profane': 'Abusive', 'Sexism': 'Gender Hate'}
    TARGET_LABELS = ['None', 'Individual', 'Organization', 'Community', 'Society']
    SEVERITY_LABELS = ['Little to None', 'Mild', 'Severe']

    # Mappings for fast lookup
    TYPE2IDX = {label: idx for idx, label in enumerate(TYPE_LABELS)}
    TARGET2IDX = {label: idx for idx, label in enumerate(TARGET_LABELS)}
    SEVERITY2IDX = {label: idx for idx, label in enumerate(SEVERITY_LABELS)}

    def __init__(self, data_path, tokenizer_name='csebuetnlp/banglabert',
                 max_length=256, max_explanation_length=128,
                 silver_explanations_path=None, normalize_text=False):
        """
        Args:
            data_path: Path to CSV or JSON file containing the dataset.
            tokenizer_name: HuggingFace tokenizer name. Default: 'csebuetnlp/banglabert'.
            max_length: Maximum token length for input text. Default: 256.
            max_explanation_length: Maximum token length for explanations. Default: 128.
            silver_explanations_path: Optional path to JSON file containing silver explanations.
            normalize_text: Whether to apply bnlp normalizer. Default: False (requires extra install).
        """
        self.max_length = max_length
        self.max_explanation_length = max_explanation_length
        self.normalize_text = normalize_text

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load data
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.df = pd.DataFrame(raw)
            self.text_col = 'comment'
        elif data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path, encoding='utf-8')
            # Detect text column name
            if 'comment' in self.df.columns:
                self.text_col = 'comment'
            elif 'text' in self.df.columns:
                self.text_col = 'text'
            else:
                self.text_col = self.df.columns[0]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Validate required columns
        required = ['type_of_hate', 'target_of_hate', 'severity_of_hate']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: '{col}' in {data_path}. "
                                 f"Available columns: {list(self.df.columns)}")

        # Load silver explanations if available
        self.explanations = {}
        if silver_explanations_path:
            with open(silver_explanations_path, 'r', encoding='utf-8') as f:
                silver = json.load(f)
                for item in silver:
                    if item.get('silver_explanation'):
                        self.explanations[item['original_idx']] = item['silver_explanation']

        # Optionally load normalizer
        self._normalizer = None
        if normalize_text:
            try:
                from normalizer import normalize as bn_normalize
                self._normalizer = bn_normalize
            except ImportError:
                print("Warning: 'normalizer' package not found. Skipping text normalization.")

        # Remap legacy labels (Profane → Abusive, Sexism → Gender Hate)
        for col in ['type_of_hate']:
            self.df[col] = self.df[col].replace(self._LABEL_REMAP)

        print(f"[BanglaHateDataset] Loaded {len(self.df)} samples from {data_path}")
        if self.explanations:
            print(f"  → {len(self.explanations)} silver explanations loaded")

    def __len__(self):
        return len(self.df)

    def _normalize(self, text):
        """Apply Bengali text normalization if available."""
        if self._normalizer:
            return self._normalizer(str(text))
        return str(text)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get and normalize text
        text = self._normalize(row[self.text_col])

        # Tokenize input text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode multi-task labels
        type_label = self.TYPE2IDX.get(row['type_of_hate'], 0)
        target_label = self.TARGET2IDX.get(row['target_of_hate'], 0)
        severity_label = self.SEVERITY2IDX.get(row['severity_of_hate'], 0)

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'type_label': torch.tensor(type_label, dtype=torch.long),
            'target_label': torch.tensor(target_label, dtype=torch.long),
            'severity_label': torch.tensor(severity_label, dtype=torch.long),
        }

        # Add explanation tokens if available (for generative head training)
        if idx in self.explanations:
            expl_text = self._normalize(self.explanations[idx])
            expl_encoding = self.tokenizer(
                expl_text,
                max_length=self.max_explanation_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['explanation_ids'] = expl_encoding['input_ids'].squeeze(0)
            item['explanation_mask'] = expl_encoding['attention_mask'].squeeze(0)
            # Create labels for autoregressive training (mask padding with -100)
            explanation_labels = expl_encoding['input_ids'].squeeze(0).clone()
            explanation_labels[expl_encoding['attention_mask'].squeeze(0) == 0] = -100
            item['explanation_labels'] = explanation_labels
            item['has_explanation'] = torch.tensor(1, dtype=torch.long)
        else:
            item['has_explanation'] = torch.tensor(0, dtype=torch.long)

        return item

    def get_class_weights(self, task='type_of_hate'):
        """
        Compute inverse-frequency class weights for a given task.
        Used as the alpha parameter in FocalLoss.

        Args:
            task: One of 'type_of_hate', 'target_of_hate', 'severity_of_hate'
        Returns:
            torch.FloatTensor of shape (num_classes,)
        """
        if task == 'type_of_hate':
            labels = self.TYPE_LABELS
        elif task == 'target_of_hate':
            labels = self.TARGET_LABELS
        elif task == 'severity_of_hate':
            labels = self.SEVERITY_LABELS
        else:
            raise ValueError(f"Unknown task: {task}")

        counts = self.df[task].value_counts()
        total = len(self.df)
        n_classes = len(labels)

        weights = []
        for label in labels:
            c = counts.get(label, 1)
            weights.append(total / (n_classes * c))

        return torch.FloatTensor(weights)
