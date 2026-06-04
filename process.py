import torch
import pandas as pd
import pickle
import os
from tqdm import tqdm
import esm
import re

PROCESSED_DATA_DIR = "./process/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading ESM-2 650M model...")
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

print(f"ESM model loaded: {esm_model.num_layers} layers, {esm_model.embed_dim} hidden dim")
print(f"Vocabulary size: {len(alphabet)}")
print(f"cls token: {alphabet.cls_idx}")
print(f"pad token: {alphabet.padding_idx}")
print(f"eos token: {alphabet.eos_idx}")
print(f"mask token: {alphabet.mask_idx}")

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')
VALID_AA_WITH_SPECIAL = set('ACDEFGHIKLMNPQRSTVWYXUBZO')


def is_valid_protein_sequence(seq):
    if not seq or not isinstance(seq, str):
        return False
    seq_upper = seq.upper()
    return all(aa in VALID_AA_WITH_SPECIAL for aa in seq_upper)


def clean_protein_sequence(seq):
    if not isinstance(seq, str):
        return None

    seq_upper = seq.upper().strip()

    seq_clean = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYXUBZO]', '', seq_upper)

    if len(seq_clean) == 0:
        return None

    return seq_clean


def tokenize_sequence(seq, alphabet):
    _, _, tokens = batch_converter([("", seq)])
    return tokens[0]


def encode_target_with_esm(target_seq, esm_model, alphabet, device):
    tokens = tokenize_sequence(target_seq, alphabet)
    tokens = tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        results = esm_model(tokens, repr_layers=[33])
        embedding = results['representations'][33]

    return embedding.squeeze(0).cpu()


def process_dataset(csv_path, split_name, esm_model, alphabet, device):
    print(f"\nProcessing {split_name} set from {csv_path}...")

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    binder_tokens_list = []
    target_embeddings_list = []
    binder_lengths = []
    target_lengths = []
    raw_binder_list = []
    raw_target_list = []

    skipped_count = 0
    invalid_samples = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        binder_seq = row['Binder']
        target_seq = row['Target']

        binder_clean = clean_protein_sequence(binder_seq)
        target_clean = clean_protein_sequence(target_seq)

        if binder_clean is None or target_clean is None:
            skipped_count += 1
            invalid_samples.append({
                'index': idx,
                'binder': binder_seq,
                'target': target_seq,
                'reason': 'Invalid characters or empty after cleaning'
            })
            continue

        if not is_valid_protein_sequence(binder_clean) or not is_valid_protein_sequence(target_clean):
            skipped_count += 1
            invalid_samples.append({
                'index': idx,
                'binder': binder_seq,
                'target': target_seq,
                'reason': 'Contains invalid amino acids'
            })
            continue

        try:
            binder_tokens = tokenize_sequence(binder_clean, alphabet)
            binder_tokens_list.append(binder_tokens)

            target_embedding = encode_target_with_esm(target_clean, esm_model, alphabet, device)
            target_embeddings_list.append(target_embedding)

            binder_lengths.append(len(binder_clean) + 2)
            target_lengths.append(len(target_clean) + 2)

            raw_binder_list.append(binder_clean)
            raw_target_list.append(target_clean)

        except Exception as e:
            skipped_count += 1
            invalid_samples.append({
                'index': idx,
                'binder': binder_seq,
                'target': target_seq,
                'reason': f'Tokenization error: {str(e)}'
            })
            continue

    if skipped_count > 0:
        print(f"\nWarning: Skipped {skipped_count} invalid samples")

        invalid_log_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}_invalid_samples.txt')
        with open(invalid_log_path, 'w') as f:
            f.write(f"Invalid samples in {split_name} set:\n")
            f.write("=" * 80 + "\n\n")
            for sample in invalid_samples:
                f.write(f"Index: {sample['index']}\n")
                f.write(f"Binder: {sample['binder']}\n")
                f.write(f"Target: {sample['target']}\n")
                f.write(f"Reason: {sample['reason']}\n")
                f.write("-" * 80 + "\n")

        print(f"Invalid samples logged to: {invalid_log_path}")

    print(f"Valid samples: {len(binder_tokens_list)}")

    if len(binder_tokens_list) == 0:
        print(f"Error: No valid samples found in {split_name} set!")
        return None

    max_binder_len = max(t.shape[0] for t in binder_tokens_list)
    max_target_len = max(t.shape[0] for t in target_embeddings_list)

    print(f"Max binder length: {max_binder_len}")
    print(f"Max target length: {max_target_len}")

    binder_input_ids = torch.full(
        (len(binder_tokens_list), max_binder_len),
        alphabet.padding_idx,
        dtype=torch.long
    )

    esm_hidden_dim = target_embeddings_list[0].shape[1]
    target_embeddings = torch.zeros(
        (len(target_embeddings_list), max_target_len, esm_hidden_dim),
        dtype=torch.float32
    )

    for i, (b_tok, t_emb) in enumerate(zip(binder_tokens_list, target_embeddings_list)):
        b_len = b_tok.shape[0]
        t_len = t_emb.shape[0]

        binder_input_ids[i, :b_len] = b_tok
        target_embeddings[i, :t_len, :] = t_emb

    binder_attention_mask = (binder_input_ids != alphabet.padding_idx).long()

    target_attention_mask = torch.zeros(
        (len(target_embeddings_list), max_target_len),
        dtype=torch.long
    )
    for i, t_emb in enumerate(target_embeddings_list):
        t_len = t_emb.shape[0]
        target_attention_mask[i, :t_len] = 1

    binder_lengths_tensor = torch.tensor(binder_lengths, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    processed_data = {
        'binder_input_ids': binder_input_ids,
        'binder_attention_mask': binder_attention_mask,
        'binder_lengths': binder_lengths_tensor,
        'target_embeddings': target_embeddings,
        'target_attention_mask': target_attention_mask,
        'target_lengths': target_lengths_tensor,
        'raw_binder': raw_binder_list,
        'raw_target': raw_target_list,
    }

    save_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Saved to {save_path}")
    print(f"Binder tokens shape: {binder_input_ids.shape}")
    print(f"Target embeddings shape: {target_embeddings.shape}")

    return processed_data


def save_esm_embedding_weights(esm_model, save_dir):
    print("\nSaving ESM embedding weights for model initialization...")

    embedding_weights = esm_model.embed_tokens.weight.detach().cpu()

    save_path = os.path.join(save_dir, 'esm_embedding_weights.pt')
    torch.save(embedding_weights, save_path)

    print(f"ESM embedding weights saved to {save_path}")
    print(f"Embedding shape: {embedding_weights.shape}")

    return embedding_weights


def main():
    train_csv = "./process/train_dataset.csv"
    val_csv = "./process/val_dataset.csv"
    test_csv = "./process/test_dataset.csv"

    process_dataset(train_csv, 'train', esm_model, alphabet, device)
    process_dataset(val_csv, 'val', esm_model, alphabet, device)
    process_dataset(test_csv, 'test', esm_model, alphabet, device)

    save_esm_embedding_weights(esm_model, PROCESSED_DATA_DIR)

    esm_embed_dim = esm_model.embed_tokens.weight.shape[1]

    alphabet_config = {
        'vocab_size': len(alphabet),
        'cls_idx': alphabet.cls_idx,
        'pad_idx': alphabet.padding_idx,
        'eos_idx': alphabet.eos_idx,
        'mask_idx': alphabet.mask_idx,
        'unk_idx': alphabet.unk_idx,
        'esm_hidden_dim': esm_model.embed_dim,
        'esm_embed_dim': esm_embed_dim,
        'aa_start_idx': 4,
        'aa_end_idx': 24,
        'model_name': 'esm2_t33_650M_UR50D',
        'num_layers': esm_model.num_layers,
    }

    with open(os.path.join(PROCESSED_DATA_DIR, 'alphabet_config.pkl'), 'wb') as f:
        pickle.dump(alphabet_config, f)

    print("\n" + "=" * 60)
    print("Data preprocessing completed!")
    print(f"Model: ESM-2 650M (33 layers)")
    print(f"Vocabulary size: {len(alphabet)}")
    print(f"ESM hidden dimension: {esm_model.embed_dim}")
    print(f"ESM embedding dimension: {esm_embed_dim}")
    print(f"Amino acid token range: 4-24")
    print(f"Data saved to: {PROCESSED_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()