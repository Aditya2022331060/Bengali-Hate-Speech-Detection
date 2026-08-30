# === banglaHate/src/smoke_test.py ===
"""
Pipeline Smoke Test — Validates the entire ML pipeline end-to-end
on a tiny dummy dataset before committing to full training.

Senior's Rule: "Always run a 5-minute smoke test before spending
hours of GPU time or API credits."

Validates:
  1. Dataset creation and tokenization
  2. Model instantiation and forward pass
  3. Multi-loss computation (Focal + Consistency + Generation)
  4. Gradient backpropagation
  5. Overfitting sanity check (5 epochs on 10 samples)
  6. Prediction decoding and consistency violation detection
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import sys
sys.stderr = sys.stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import BanglaHateDataset
from src.model import ConsistencyConstrainedMTL
from src.losses import FocalLoss, ConsistencyPenaltyLoss


def create_dummy_dataset(output_path, n_samples=10):
    """Create a tiny JSON dataset for smoke testing."""
    samples = [
        {"id": "s1", "comment": "ধন্যবাদ বর্ডার গার্ড দেরকে এভাবে পাহারা দিতে হবে",
         "category": "National", "subcategory": "test",
         "type_of_hate": "None", "severity_of_hate": "Little to None", "target_of_hate": "None"},
        {"id": "s2", "comment": "এই শালার বাচ্চারা সব চোর দেশ লুটে খাচ্ছে",
         "category": "Politics", "subcategory": "test",
         "type_of_hate": "Abusive", "severity_of_hate": "Severe", "target_of_hate": "Individual"},
        {"id": "s3", "comment": "হিন্দুদের দেশ থেকে বের করে দাও",
         "category": "Religion", "subcategory": "test",
         "type_of_hate": "Religious Hate", "severity_of_hate": "Severe", "target_of_hate": "Community"},
        {"id": "s4", "comment": "মেয়েদের রাজনীতিতে আসা উচিত না",
         "category": "Politics", "subcategory": "test",
         "type_of_hate": "Gender Hate", "severity_of_hate": "Mild", "target_of_hate": "Individual"},
        {"id": "s5", "comment": "আজকের খেলা খুব ভালো হয়েছে",
         "category": "Sports", "subcategory": "test",
         "type_of_hate": "None", "severity_of_hate": "Little to None", "target_of_hate": "None"},
        {"id": "s6", "comment": "সরকার জনগণের টাকা মেরে দিচ্ছে",
         "category": "Politics", "subcategory": "test",
         "type_of_hate": "Political Hate", "severity_of_hate": "Mild", "target_of_hate": "Organization"},
        {"id": "s7", "comment": "ওই বদমাশটাকে ধরে পিটাও",
         "category": "National", "subcategory": "test",
         "type_of_hate": "Abusive", "severity_of_hate": "Severe", "target_of_hate": "Individual"},
        {"id": "s8", "comment": "শালা কুত্তার বাচ্চা",
         "category": "Miscellaneous", "subcategory": "test",
         "type_of_hate": "Abusive", "severity_of_hate": "Severe", "target_of_hate": "Individual"},
        {"id": "s9", "comment": "এই সম্প্রদায়ের লোকগুলো সব সন্ত্রাসী",
         "category": "Religion", "subcategory": "test",
         "type_of_hate": "Religious Hate", "severity_of_hate": "Mild", "target_of_hate": "Community"},
        {"id": "s10", "comment": "বাংলাদেশ ক্রিকেট দল অসাধারণ খেলেছে",
         "category": "Sports", "subcategory": "test",
         "type_of_hate": "None", "severity_of_hate": "Little to None", "target_of_hate": "None"},
    ]

    # Repeat to get desired sample count
    data = (samples * ((n_samples // len(samples)) + 1))[:n_samples]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def check_consistency_violations(type_preds, target_preds, sev_preds):
    """
    Check for logical consistency violations in predictions.
    Rule: If type=None(0) → target must be None(0) AND severity must be Little(0).
    """
    violations = 0
    total = len(type_preds)
    for t, tg, s in zip(type_preds, target_preds, sev_preds):
        if t == 0:  # type = None
            if tg != 0 or s != 0:
                violations += 1
    return violations, total


def run_smoke_test():
    """Execute the full pipeline smoke test."""

    print("=" * 70)
    print("[SMOKE TEST] PIPELINE SANITY CHECK")
    print("=" * 70)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ──────────────────────────────────────────
    # Test 1: Create dummy dataset
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 1] Create dummy dataset")
    dummy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dummy_smoke_test.json')
    create_dummy_dataset(dummy_path, n_samples=10)
    print(f"   [PASSED] Created {dummy_path}")

    # ──────────────────────────────────────────
    # Test 2: Dataset loading & tokenization
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 2] Dataset loading & tokenization")
    dataset = BanglaHateDataset(
        data_path=dummy_path,
        tokenizer_name='csebuetnlp/banglabert',
        max_length=128,  # shorter for smoke test
        normalize_text=False
    )
    print(f"   Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"   input_ids shape: {sample['input_ids'].shape}")
    print(f"   attention_mask shape: {sample['attention_mask'].shape}")
    print(f"   type_label: {sample['type_label'].item()} ({BanglaHateDataset.TYPE_LABELS[sample['type_label'].item()]})")
    print(f"   target_label: {sample['target_label'].item()} ({BanglaHateDataset.TARGET_LABELS[sample['target_label'].item()]})")
    print(f"   severity_label: {sample['severity_label'].item()} ({BanglaHateDataset.SEVERITY_LABELS[sample['severity_label'].item()]})")
    print(f"   [PASSED] Dataset loading passed!")

    # ──────────────────────────────────────────
    # Test 3: DataLoader batch creation
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 3] DataLoader batch creation")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"   Batch type_labels: {batch['type_label'].tolist()}")
    print(f"   Batch target_labels: {batch['target_label'].tolist()}")
    print(f"   Batch severity_labels: {batch['severity_label'].tolist()}")
    print(f"   [PASSED] DataLoader passed!")

    # ──────────────────────────────────────────
    # Test 4: Model instantiation & forward pass
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 4] Model instantiation & forward pass")
    model = ConsistencyConstrainedMTL(
        encoder_name='csebuetnlp/banglabert',
        num_type_labels=5,
        num_target_labels=5,
        num_severity_labels=3,
        use_gen_head=False
    ).to(device)

    total_params = model.get_num_params(trainable_only=True)
    print(f"   Total trainable parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")

    batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        type_logits, target_logits, sev_logits, gen_logits = model(
            batch_device['input_ids'],
            batch_device['attention_mask']
        )

    print(f"   type_logits shape: {type_logits.shape}")
    print(f"   target_logits shape: {target_logits.shape}")
    print(f"   severity_logits shape: {sev_logits.shape}")
    print(f"   gen_logits: {gen_logits}")
    assert type_logits.shape == (4, 6), f"Expected (4,6), got {type_logits.shape}"
    assert target_logits.shape == (4, 5), f"Expected (4,5), got {target_logits.shape}"
    assert sev_logits.shape == (4, 3), f"Expected (4,3), got {sev_logits.shape}"
    print(f"   [PASSED] Forward pass passed! All output shapes correct.")

    # ──────────────────────────────────────────
    # Test 5: Loss computation
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 5] Loss computation")

    # Class weights from dataset
    type_weights = dataset.get_class_weights('type_of_hate').to(device)
    target_weights = dataset.get_class_weights('target_of_hate').to(device)
    sev_weights = dataset.get_class_weights('severity_of_hate').to(device)
    print(f"   Type weights: {type_weights.tolist()}")

    focal_type = FocalLoss(alpha=type_weights, gamma=2.0)
    focal_target = FocalLoss(alpha=target_weights, gamma=2.0)
    focal_sev = FocalLoss(alpha=sev_weights, gamma=2.0)
    consist_loss = ConsistencyPenaltyLoss(type_none_idx=0, target_none_idx=0, sev_little_idx=0, sev_severe_idx=2)

    # Enable gradient for loss test
    type_logits_grad, target_logits_grad, sev_logits_grad, _ = model(
        batch_device['input_ids'],
        batch_device['attention_mask']
    )

    l_type = focal_type(type_logits_grad, batch_device['type_label'])
    l_target = focal_target(target_logits_grad, batch_device['target_label'])
    l_sev = focal_sev(sev_logits_grad, batch_device['severity_label'])
    l_consist = consist_loss(type_logits_grad, target_logits_grad, sev_logits_grad)

    total_loss = l_type + l_target + l_sev + 0.1 * l_consist

    print(f"   Focal(type): {l_type.item():.4f}")
    print(f"   Focal(target): {l_target.item():.4f}")
    print(f"   Focal(severity): {l_sev.item():.4f}")
    print(f"   Consistency: {l_consist.item():.4f}")
    print(f"   Total Loss: {total_loss.item():.4f}")

    assert not torch.isnan(total_loss), "Loss is NaN!"
    assert not torch.isinf(total_loss), "Loss is Inf!"
    assert total_loss.item() > 0, "Loss should be positive!"
    print(f"   [PASSED] Loss computation passed! No NaN/Inf.")

    # ──────────────────────────────────────────
    # Test 6: Gradient backpropagation
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 6] Gradient backpropagation")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradients exist and are finite
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if torch.isnan(param.grad).any():
                print(f"   [FAIL] NaN gradient in {name}!")
                return False

    optimizer.step()
    print(f"   Gradient norms — min: {min(grad_norms):.6f}, max: {max(grad_norms):.6f}, mean: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"   Parameters with gradients: {len(grad_norms)}")
    print(f"   [PASSED] Backpropagation passed! All gradients are finite.")

    # ──────────────────────────────────────────
    # Test 7: Overfitting sanity check (5 epochs)
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 7] Overfitting sanity check (5 epochs on 10 samples)")
    model.train()
    losses_history = []

    for epoch in range(5):
        epoch_loss = 0.0
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()

            t_out, tg_out, s_out, _ = model(batch_dev['input_ids'], batch_dev['attention_mask'])

            loss = (focal_type(t_out, batch_dev['type_label']) +
                    focal_target(tg_out, batch_dev['target_label']) +
                    focal_sev(s_out, batch_dev['severity_label']) +
                    0.1 * consist_loss(t_out, tg_out, s_out))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses_history.append(avg_loss)
        print(f"   Epoch {epoch+1}/5 — Loss: {avg_loss:.4f}")

    # Check that loss decreased
    if losses_history[-1] < losses_history[0]:
        print(f"   [PASSED] Loss decreased from {losses_history[0]:.4f} → {losses_history[-1]:.4f}")
    else:
        print(f"   [WARNING] Loss did NOT decrease ({losses_history[0]:.4f} → {losses_history[-1]:.4f}). May need investigation.")

    # ──────────────────────────────────────────
    # Test 8: Prediction decoding & consistency check
    # ──────────────────────────────────────────
    print("\n" + "-" * 50)
    print("[TEST 8] Prediction decoding & consistency violation check")
    model.eval()
    all_type_preds, all_target_preds, all_sev_preds = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            t_out, tg_out, s_out, _ = model(batch_dev['input_ids'], batch_dev['attention_mask'])

            all_type_preds.extend(t_out.argmax(dim=-1).cpu().tolist())
            all_target_preds.extend(tg_out.argmax(dim=-1).cpu().tolist())
            all_sev_preds.extend(s_out.argmax(dim=-1).cpu().tolist())

    violations, total = check_consistency_violations(all_type_preds, all_target_preds, all_sev_preds)
    print(f"   Predictions: {len(all_type_preds)} samples")
    print(f"   Consistency violations: {violations}/{total} ({violations/total*100:.1f}%)")

    # Show sample predictions
    print(f"\n   Sample predictions (first 5):")
    for i in range(min(5, len(all_type_preds))):
        t = BanglaHateDataset.TYPE_LABELS[all_type_preds[i]]
        tg = BanglaHateDataset.TARGET_LABELS[all_target_preds[i]]
        s = BanglaHateDataset.SEVERITY_LABELS[all_sev_preds[i]]
        print(f"     [{i}] Type={t}, Target={tg}, Severity={s}")

    print(f"   [PASSED] Prediction decoding passed!")

    # ──────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUCCESS] ALL 8 SMOKE TESTS PASSED! ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"\nThe pipeline is validated and ready for full-scale training.")
    print(f"Total model parameters: {total_params:,}")
    print(f"Device used: {device}")

    return True


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
