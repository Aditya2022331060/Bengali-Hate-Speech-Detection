# === banglaHate/src/model.py ===
"""
ConsistencyConstrainedMTL — Multi-Task Learning model for Bengali Hate Speech
classification with consistency constraints and optional generative explanation head.

Architecture:
  ┌─────────────────────────┐
  │  BanglaBERT Encoder     │  (csebuetnlp/banglabert, 110M params)
  │  [CLS] pooled output    │
  └──────────┬──────────────┘
             │ h_cls (768-dim)
      ┌──────┼──────┬──────┐
      ▼      ▼      ▼      ▼
  ┌──────┐┌──────┐┌──────┐┌──────────┐
  │Type  ││Target││ Sev  ││ Gen Head │  (optional)
  │Head  ││ Head ││ Head ││ (Decoder)│
  │→ 5   ││→ 5   ││→ 3   ││→ vocab  │
  └──────┘└──────┘└──────┘└──────────┘

Each classification head: Linear(768, hidden) → ReLU → Dropout → Linear(hidden, n_classes)
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ClassificationHead(nn.Module):
    """
    Two-layer classification head with dropout.

    Args:
        input_dim: Dimension of input features (768 for BanglaBERT).
        hidden_dim: Hidden layer dimension. Default: 256.
        num_classes: Number of output classes.
        dropout: Dropout probability. Default: 0.3.
    """

    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class ConsistencyConstrainedMTL(nn.Module):
    """
    Multi-Task Learning model with 3 classification heads and an optional
    generative explanation decoder.

    The model uses BanglaBERT as a shared encoder and routes the [CLS]
    representation to task-specific heads. The consistency constraint is
    applied at the loss level (not in the architecture), keeping the model
    architecture clean.

    Args:
        encoder_name: HuggingFace model name for the shared encoder.
                      Default: 'csebuetnlp/banglabert'.
        num_type_labels: Number of hate type classes. Default: 5.
        num_target_labels: Number of target classes. Default: 5.
        num_severity_labels: Number of severity classes. Default: 3.
        hidden_dim: Hidden dimension in classification heads. Default: 256.
        dropout: Dropout rate. Default: 0.3.
        use_gen_head: Whether to include the generative explanation head.
                      Default: False.
    """

    def __init__(self, encoder_name='csebuetnlp/banglabert',
                 num_type_labels=5, num_target_labels=5, num_severity_labels=3,
                 hidden_dim=256, dropout=0.3, use_gen_head=False):
        super().__init__()

        # Shared encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size  # 768 for BanglaBERT

        # Task-specific classification heads
        self.type_head = ClassificationHead(encoder_dim, num_type_labels, hidden_dim, dropout)
        self.target_head = ClassificationHead(encoder_dim, num_target_labels, hidden_dim, dropout)
        self.severity_head = ClassificationHead(encoder_dim, num_severity_labels, hidden_dim, dropout)

        # Optional generative head (simple linear projection to vocabulary)
        self.use_gen_head = use_gen_head
        if use_gen_head:
            vocab_size = self.encoder.config.vocab_size
            self.gen_projection = nn.Linear(encoder_dim, vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            decoder_input_ids: Optional explanation token IDs for generative head.
                               Shape: (batch_size, expl_seq_len)

        Returns:
            type_logits: Shape (batch_size, num_type_labels)
            target_logits: Shape (batch_size, num_target_labels)
            severity_logits: Shape (batch_size, num_severity_labels)
            gen_logits: Shape (batch_size, expl_seq_len, vocab_size) or None
        """
        # Encode input through BanglaBERT
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]  # (batch, 768)

        # Classification heads
        type_logits = self.type_head(cls_hidden)
        target_logits = self.target_head(cls_hidden)
        severity_logits = self.severity_head(cls_hidden)

        # Generative head (optional)
        gen_logits = None
        if self.use_gen_head and decoder_input_ids is not None:
            # Simple approach: use encoder to process explanation tokens too
            # and project to vocabulary for autoregressive generation
            decoder_output = self.encoder(
                input_ids=decoder_input_ids,
                attention_mask=(decoder_input_ids != self.encoder.config.pad_token_id).long()
            )
            gen_logits = self.gen_projection(decoder_output.last_hidden_state)

        return type_logits, target_logits, severity_logits, gen_logits

    def get_num_params(self, trainable_only=True):
        """Return total number of (trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
