"""
Conditional Peptide Generation Inference Script
Generate peptide sequences conditioned on target protein sequences.

Usage:
    python inference_demo.py
"""

import torch
import numpy as np
import esm
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from model import ConditionalCNNModel


class ConditionalConstrainedWrapper:
    """Wrapper to apply length constraints during generation."""

    def __init__(self, model, target_lengths, alphabet_config):
        self.model = model
        self.target_lengths = target_lengths
        self.alphabet_config = alphabet_config

    def __call__(self, x, t, target_embeddings, target_mask):
        logits = self.model(x, t, target_embeddings, target_mask)
        logits = self.apply_length_constraint(logits)
        return torch.softmax(logits, dim=-1)

    def apply_length_constraint(self, logits):
        """Apply hard length constraints to logits."""
        batch_size, seq_len, vocab_size = logits.shape
        cls_idx = self.alphabet_config['cls_idx']
        eos_idx = self.alphabet_config['eos_idx']
        pad_idx = self.alphabet_config['pad_idx']

        NEG = -1e9
        POS = 1e9

        for i in range(batch_size):
            L = int(self.target_lengths[i].item())
            L = min(L, seq_len)
            if L <= 0:
                continue

            # Position 0 → CLS
            logits[i, 0, :] = NEG
            logits[i, 0, cls_idx] = POS

            # Position L-1 → EOS
            if L > 1:
                logits[i, L - 1, :] = NEG
                logits[i, L - 1, eos_idx] = POS

            # Position L: → PAD
            if L < seq_len:
                logits[i, L:, :] = NEG
                logits[i, L:, pad_idx] = POS

            # Middle positions: forbid special tokens
            if L > 2:
                logits[i, 1:L - 1, cls_idx] = NEG
                logits[i, 1:L - 1, eos_idx] = NEG
                logits[i, 1:L - 1, pad_idx] = NEG

        return logits


class PeptideGenerator:
    """Conditional peptide generator using flow matching with length constraints."""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        # Load ESM model
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.eval().to(device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.alphabet_config = checkpoint['alphabet_config']
        config = checkpoint['config']

        # Build model
        esm_embedding_weights = self.esm_model.embed_tokens.weight.detach().cpu()
        self.model = ConditionalCNNModel(
            vocab_size=self.alphabet_config['vocab_size'],
            esm_embed_dim=self.alphabet_config['esm_embed_dim'],
            esm_hidden_dim=self.alphabet_config['esm_hidden_dim'],
            embed_dim=config.get('embed_dim', 512),
            hidden_dim=config.get('hidden_dim', 512),
            cross_attn_heads=config.get('cross_attn_heads', 8),
            cross_attn_layers=config.get('cross_attn_layers', 3),
            kan_num_grids=8,
            kan_grid_range=(-2., 2.),
            esm_embedding_weights=esm_embedding_weights,
            finetune_embedding=True
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def encode_target(self, target_sequence):
        """Encode target protein sequence using ESM-2."""
        _, _, tokens = self.batch_converter([("target", target_sequence)])
        tokens = tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[33])
            embedding = results['representations'][33]

        mask = (tokens != self.alphabet.padding_idx).long()
        return embedding.squeeze(0), mask.squeeze(0)

    def create_initial_noise(self, batch_size, seq_length):
        """Create initial masked noise for generation."""
        cls_idx = self.alphabet_config['cls_idx']
        eos_idx = self.alphabet_config['eos_idx']
        mask_idx = self.alphabet_config['mask_idx']

        x_init = torch.full((batch_size, seq_length), mask_idx, dtype=torch.long, device=self.device)
        x_init[:, 0] = cls_idx
        x_init[:, seq_length - 1] = eos_idx
        return x_init

    def decode_sequence(self, token_ids, expected_length):
        """Decode token IDs to amino acid sequence."""
        aa_start = self.alphabet_config['aa_start_idx']
        aa_end = self.alphabet_config['aa_end_idx']

        if torch.is_tensor(token_ids):
            token_ids = token_ids.cpu().tolist()

        sequence = ""
        for tok in token_ids[1:expected_length + 1]:
            if aa_start <= tok <= aa_end:
                sequence += self.alphabet.get_tok(tok)
        return sequence

    def generate(self, target_sequence, peptide_length, n_samples=5, steps=50):
        """Generate peptide sequences conditioned on target protein with length constraints."""
        target_emb, target_mask = self.encode_target(target_sequence)
        target_emb_batch = target_emb.unsqueeze(0).repeat(n_samples, 1, 1)
        target_mask_batch = target_mask.unsqueeze(0).repeat(n_samples, 1)

        seq_length = peptide_length + 2
        x_init = self.create_initial_noise(n_samples, seq_length)

        # Apply length constraints during generation
        target_lengths = torch.tensor([seq_length] * n_samples, device=self.device)
        constrained_wrapper = ConditionalConstrainedWrapper(
            self.model,
            target_lengths=target_lengths,
            alphabet_config=self.alphabet_config
        )

        # Setup solver with constrained wrapper
        scheduler = PolynomialConvexScheduler(n=2.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        solver = MixtureDiscreteEulerSolver(
            model=constrained_wrapper,
            path=path,
            vocabulary_size=self.alphabet_config['vocab_size']
        )

        with torch.no_grad():
            generated = solver.sample(
                x_init=x_init,
                step_size=1.0 / steps,
                verbose=False,
                time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=self.device),
                target_embeddings=target_emb_batch,
                target_mask=target_mask_batch
            )

        # Enforce special tokens
        cls_idx = self.alphabet_config['cls_idx']
        eos_idx = self.alphabet_config['eos_idx']
        pad_idx = self.alphabet_config['pad_idx']
        generated[:, 0] = cls_idx
        generated[:, seq_length - 1] = eos_idx
        if seq_length < generated.shape[1]:
            generated[:, seq_length:] = pad_idx

        sequences = [self.decode_sequence(generated[i], peptide_length) for i in range(n_samples)]
        return sequences


def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    CHECKPOINT_PATH = "./ckpt/best_loss_model.ckpt"
    TARGET_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    PEPTIDE_LENGTH = 15
    N_SAMPLES = 1
    STEPS = 150

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize generator
    generator = PeptideGenerator(checkpoint_path=CHECKPOINT_PATH, device=device)

    # Generate peptides with length constraints
    sequences = generator.generate(
        target_sequence=TARGET_SEQUENCE,
        peptide_length=PEPTIDE_LENGTH,
        n_samples=N_SAMPLES,
        steps=STEPS
    )

    # Display results
    print(f"\nGenerated {len(sequences)} peptides (length={PEPTIDE_LENGTH}):")
    print("-" * 50)
    for i, seq in enumerate(sequences, 1):
        print(f"{i:2d}. {seq} (len={len(seq)})")
    print("-" * 50)


if __name__ == "__main__":
    main()