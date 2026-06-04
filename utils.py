import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def plot_training_curves(training_log, save_path):
    epochs_list = training_log['epochs']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs_list, training_log['train_loss'], label='Train Total Loss', linewidth=2)
    axes[0, 0].plot(epochs_list, training_log['val_loss'], label='Val Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss (Flow + CE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_list, training_log['train_flow_loss'], label='Train Flow Loss', linewidth=2)
    axes[0, 1].plot(epochs_list, training_log['val_flow_loss'], label='Val Flow Loss', linewidth=2)
    axes[0, 1].plot(epochs_list, training_log['train_ce_loss'], label='Train CE Loss',
                    linewidth=2, linestyle='--')
    axes[0, 1].plot(epochs_list, training_log['val_ce_loss'], label='Val CE Loss',
                    linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Flow Loss vs CE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs_list, training_log['train_perplexity'], label='Train Perplexity', linewidth=2)
    axes[0, 2].plot(epochs_list, training_log['val_perplexity'], label='Val Perplexity', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Perplexity')
    axes[0, 2].set_title('Training and Validation Perplexity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_list, training_log['train_aar'], label='Train AAR', linewidth=2)
    axes[1, 0].plot(epochs_list, training_log['val_aar'], label='Val AAR', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AAR')
    axes[1, 0].set_title('Training and Validation AAR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_list, training_log['learning_rates'], label='Learning Rate',
                    linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs_list, training_log['test_loss'], label='Test Loss',
                    linewidth=2, color='green')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Test Loss')
    axes[1, 2].set_title('Test Loss per Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_mask_noise(reference_sequences, lengths, alphabet_config, device='cuda'):
    batch_size, max_len = reference_sequences.shape

    mask_idx = alphabet_config['mask_idx']
    cls_idx = alphabet_config['cls_idx']
    eos_idx = alphabet_config['eos_idx']
    pad_idx = alphabet_config['pad_idx']

    x_0 = torch.full((batch_size, max_len), mask_idx, dtype=torch.long, device=device)

    for i, length in enumerate(lengths):
        length = length.item() if torch.is_tensor(length) else length
        length = min(length, max_len)

        if length >= 1:
            x_0[i, 0] = cls_idx
        if length >= 2:
            x_0[i, length - 1] = eos_idx
        if length < max_len:
            x_0[i, length:] = pad_idx

    return x_0


def create_uniform_noise(reference_sequences, lengths, alphabet_config, device='cuda'):
    batch_size, max_len = reference_sequences.shape

    aa_start = alphabet_config['aa_start_idx']
    aa_end = alphabet_config['aa_end_idx']
    cls_idx = alphabet_config['cls_idx']
    eos_idx = alphabet_config['eos_idx']
    pad_idx = alphabet_config['pad_idx']

    x_0 = torch.randint(aa_start, aa_end + 1, (batch_size, max_len), device=device)

    for i, length in enumerate(lengths):
        length = length.item() if torch.is_tensor(length) else length
        length = min(length, max_len)

        if length >= 1:
            x_0[i, 0] = cls_idx
        if length >= 2:
            x_0[i, length - 1] = eos_idx
        if length < max_len:
            x_0[i, length:] = pad_idx

    return x_0


def apply_length_constraint_masking(logits, target_lengths, alphabet_config):
    batch_size, seq_len, vocab_size = logits.shape

    cls_idx = alphabet_config['cls_idx']
    eos_idx = alphabet_config['eos_idx']
    pad_idx = alphabet_config['pad_idx']

    LARGE_NEGATIVE = -1e10
    LARGE_POSITIVE = 1e10

    for i in range(batch_size):
        target_len = target_lengths[i].item() if torch.is_tensor(target_lengths[i]) else target_lengths[i]
        target_len = min(target_len, seq_len)

        if target_len <= 0:
            continue

        logits[i, 0, :] = LARGE_NEGATIVE
        logits[i, 0, cls_idx] = LARGE_POSITIVE

        if target_len > 1:
            eos_pos = target_len - 1
            logits[i, eos_pos, :] = LARGE_NEGATIVE
            logits[i, eos_pos, eos_idx] = LARGE_POSITIVE

        if target_len < seq_len:
            logits[i, target_len:, :] = LARGE_NEGATIVE
            logits[i, target_len:, pad_idx] = LARGE_POSITIVE

        if target_len > 2:
            logits[i, 1:target_len - 1, cls_idx] = LARGE_NEGATIVE
            logits[i, 1:target_len - 1, eos_idx] = LARGE_NEGATIVE
            logits[i, 1:target_len - 1, pad_idx] = LARGE_NEGATIVE

    return logits


def verify_sequence_constraints(sequences, target_lengths, alphabet_config):
    errors = []

    cls_idx = alphabet_config['cls_idx']
    eos_idx = alphabet_config['eos_idx']
    pad_idx = alphabet_config['pad_idx']

    for i in range(sequences.shape[0]):
        target_len = target_lengths[i].item() if torch.is_tensor(target_lengths[i]) else target_lengths[i]
        seq = sequences[i]

        if seq[0] != cls_idx:
            errors.append(f"Sample {i}: CLS error (got {seq[0]}, expected {cls_idx})")

        if target_len > 1 and seq[target_len - 1] != eos_idx:
            errors.append(
                f"Sample {i}: EOS error at pos {target_len - 1} (got {seq[target_len - 1]}, expected {eos_idx})")

        if target_len < len(seq):
            pad_region = seq[target_len:]
            if not (pad_region == pad_idx).all():
                wrong_pads = (pad_region != pad_idx).sum().item()
                errors.append(f"Sample {i}: PAD error ({wrong_pads} wrong tokens in PAD region)")

        if target_len > 2:
            middle_region = seq[1:target_len - 1]
            if (middle_region == cls_idx).any() or (middle_region == eos_idx).any() or (middle_region == pad_idx).any():
                errors.append(f"Sample {i}: Special token in middle region")

    return errors


def compute_ce_loss(logits, targets, lengths, alphabet_config, device):
    batch_size, seq_len, vocab_size = logits.shape

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        length = length.item() if torch.is_tensor(length) else length
        length = min(length, seq_len)
        if length > 0:
            mask[i, :length] = True

    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_flat = logits[mask]
    targets_flat = targets[mask]

    return F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)


def calculate_perplexity(logits, targets, lengths, alphabet_config):
    batch_size = logits.shape[0]
    total_log_prob = 0.0
    total_tokens = 0

    aa_start = alphabet_config['aa_start_idx']
    aa_end = alphabet_config['aa_end_idx']

    log_probs = F.log_softmax(logits, dim=-1)

    for i in range(batch_size):
        length = lengths[i].item() if torch.is_tensor(lengths[i]) else lengths[i]
        length = min(length, logits.shape[1])

        if length > 2:
            for pos in range(1, length - 1):
                target_token = targets[i, pos]
                if aa_start <= target_token <= aa_end:
                    total_log_prob += log_probs[i, pos, target_token].item()
                    total_tokens += 1

    if total_tokens == 0:
        return float('inf')

    avg_log_prob = total_log_prob / total_tokens
    perplexity = np.exp(-avg_log_prob)

    return perplexity


def calculate_aar(predictions, targets, lengths):
    batch_size = predictions.shape[0]
    correct_aa = 0
    total_aa = 0

    for i in range(batch_size):
        length = lengths[i].item() if torch.is_tensor(lengths[i]) else lengths[i]
        length = min(length, predictions.shape[1])

        if length > 2:
            aa_pred = predictions[i, 1:length - 1]
            aa_target = targets[i, 1:length - 1]
            correct_aa += (aa_pred == aa_target).sum().item()
            total_aa += (length - 2)

    aar = correct_aa / total_aa if total_aa > 0 else 0.0
    return aar