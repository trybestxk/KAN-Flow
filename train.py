import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import random

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from model import ConditionalCNNModel, count_parameters
from loder import get_all_dataloaders
from utils import (
    EarlyStopping,
    create_mask_noise,
    create_uniform_noise,
    apply_length_constraint_masking,
    verify_sequence_constraints,
    compute_ce_loss,
    calculate_perplexity,
    calculate_aar
)

random.seed(200)
torch.manual_seed(200)
np.random.seed(200)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(200)

lr = 1e-4
epochs = 100
embed_dim = 512
hidden_dim = 512
epsilon = 1e-3
batch_size = 64
num_workers = 4
cross_attn_layers = 3
cross_attn_heads = 8
ce_weight = 1.0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

NOISE_TYPE = 'mask'

PROCESSED_DATA_DIR = "./process/"
CHECKPOINT_DIR = f"./ckpt/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_step(model, loss_fn, path, batch, alphabet_config, device, noise_type='mask'):
    x_1 = batch['binder_input_ids'].to(device)
    binder_lengths = batch['binder_lengths'].to(device)
    target_embeddings = batch['target_embeddings'].to(device)
    target_mask = batch['target_attention_mask'].to(device)

    batch_size = x_1.shape[0]

    if noise_type == 'mask':
        x_0 = create_mask_noise(x_1, binder_lengths, alphabet_config, device)
    else:
        x_0 = create_uniform_noise(x_1, binder_lengths, alphabet_config, device)

    t = torch.rand(batch_size, device=device) * (1 - epsilon)
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    logits = model(
        x=path_sample.x_t,
        t=path_sample.t,
        target_embeddings=target_embeddings,
        target_mask=target_mask
    )

    ce_loss = compute_ce_loss(logits, x_1, binder_lengths, alphabet_config, device)

    logits_masked = apply_length_constraint_masking(logits.clone(), binder_lengths, alphabet_config)

    flow_loss = loss_fn(
        logits=logits_masked,
        x_1=x_1,
        x_t=path_sample.x_t,
        t=path_sample.t
    )

    total_loss = flow_loss + ce_weight * ce_loss

    prediction = logits_masked.argmax(dim=-1)
    perplexity = calculate_perplexity(logits, x_1, binder_lengths, alphabet_config)

    return total_loss, flow_loss, ce_loss, prediction, x_1, binder_lengths, perplexity


def train_epoch(model, train_loader, optimizer, loss_fn, path, alphabet_config, epoch, noise_type='mask'):
    model.train()

    all_predictions = []
    all_targets = []
    all_lengths = []
    train_total_losses = []
    train_flow_losses = []
    train_ce_losses = []
    train_perplexities = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [{noise_type} noise]')
    for batch in pbar:
        optimizer.zero_grad()

        total_loss, flow_loss, ce_loss, prediction, target, lengths, perplexity = train_step(
            model, loss_fn, path, batch, alphabet_config, device, noise_type
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_total_losses.append(total_loss.item())
        train_flow_losses.append(flow_loss.item())
        train_ce_losses.append(ce_loss.item())
        train_perplexities.append(perplexity)
        all_predictions.append(prediction.cpu())
        all_targets.append(target.cpu())
        all_lengths.extend(lengths.cpu().tolist())

        pbar.set_postfix({
            'Total': f'{total_loss.item():.4f}',
            'Flow': f'{flow_loss.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'PPL': f'{perplexity:.2f}'
        })

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_lengths = torch.tensor(all_lengths)

    errors = verify_sequence_constraints(all_predictions, all_lengths, alphabet_config)
    if errors and len(errors) > 0:
        print(f"\n  [WARNING] Found {len(errors)} constraint violations in training predictions")
        for err in errors[:3]:
            print(f"    {err}")

    aar = calculate_aar(all_predictions, all_targets, all_lengths)
    avg_total_loss = np.mean(train_total_losses)
    avg_flow_loss = np.mean(train_flow_losses)
    avg_ce_loss = np.mean(train_ce_losses)
    avg_perplexity = np.mean(train_perplexities)

    return avg_total_loss, avg_flow_loss, avg_ce_loss, aar, avg_perplexity


@torch.no_grad()
def evaluate_epoch(model, data_loader, loss_fn, path, alphabet_config, epoch, split_name="Val", noise_type='mask'):
    model.eval()

    all_predictions = []
    all_targets = []
    all_lengths = []
    eval_total_losses = []
    eval_flow_losses = []
    eval_ce_losses = []
    eval_perplexities = []

    for batch in tqdm(data_loader, desc=f'{split_name} Epoch {epoch + 1} [{noise_type} noise]'):
        total_loss, flow_loss, ce_loss, prediction, target, lengths, perplexity = train_step(
            model, loss_fn, path, batch, alphabet_config, device, noise_type
        )

        eval_total_losses.append(total_loss.item())
        eval_flow_losses.append(flow_loss.item())
        eval_ce_losses.append(ce_loss.item())
        eval_perplexities.append(perplexity)
        all_predictions.append(prediction.cpu())
        all_targets.append(target.cpu())
        all_lengths.extend(lengths.cpu().tolist())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_lengths = torch.tensor(all_lengths)

    errors = verify_sequence_constraints(all_predictions, all_lengths, alphabet_config)
    if errors and len(errors) > 0:
        print(f"\n  [WARNING] Found {len(errors)} constraint violations in {split_name} predictions")
        for err in errors[:3]:
            print(f"    {err}")

    aar = calculate_aar(all_predictions, all_targets, all_lengths)
    avg_total_loss = np.mean(eval_total_losses)
    avg_flow_loss = np.mean(eval_flow_losses)
    avg_ce_loss = np.mean(eval_ce_losses)
    avg_perplexity = np.mean(eval_perplexities)

    return avg_total_loss, avg_flow_loss, avg_ce_loss, aar, avg_perplexity


def build_model(alphabet_config, esm_embedding_weights):
    model = ConditionalCNNModel(
        vocab_size=alphabet_config['vocab_size'],
        esm_embed_dim=alphabet_config['esm_embed_dim'],
        esm_hidden_dim=alphabet_config['esm_hidden_dim'],
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        cross_attn_heads=cross_attn_heads,
        cross_attn_layers=cross_attn_layers,
        kan_num_grids=8,
        kan_grid_range=(-2., 2.),
        esm_embedding_weights=esm_embedding_weights,
        finetune_embedding=True
    ).to(device)
    return model


def save_checkpoint(save_path, model, optimizer, scheduler_lr, epoch,
                    val_loss, val_aar, val_perplexity, alphabet_config, save_reason):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_lr.state_dict(),
        'val_loss': val_loss,
        'val_aar': val_aar,
        'val_perplexity': val_perplexity,
        'alphabet_config': alphabet_config,
        'noise_type': NOISE_TYPE,
        'save_reason': save_reason,
        'config': {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'cross_attn_heads': cross_attn_heads,
            'cross_attn_layers': cross_attn_layers,
            'ce_weight': ce_weight,
            'lr': lr,
            'batch_size': batch_size,
        }
    }, save_path)


def main():
    print("=" * 60)
    print(f"Conditional Peptide Generation with ESM-2 650M Embedding")
    print(f"Noise Type:          {NOISE_TYPE.upper()}")
    print(f"Cross Attn Layers:   {cross_attn_layers}")
    print(f"Cross Attn Heads:    {cross_attn_heads}")
    print(f"CE Loss Weight:      {ce_weight}")
    print(f"Model Embed Dim:     {embed_dim}")
    print(f"Model Hidden Dim:    {hidden_dim}")
    print(f"Batch Size:          {batch_size}")
    print(f"Length Constraint:   ENABLED (Logits Masking)")
    print("=" * 60)

    print("\nLoading alphabet config...")
    with open(os.path.join(PROCESSED_DATA_DIR, 'alphabet_config.pkl'), 'rb') as f:
        alphabet_config = pickle.load(f)

    print("Alphabet configuration:")
    for k, v in alphabet_config.items():
        print(f"  {k}: {v}")

    print("\nLoading ESM embedding weights...")
    esm_embedding_weights = torch.load(
        os.path.join(PROCESSED_DATA_DIR, 'esm_embedding_weights.pt')
    )
    print(f"ESM embedding shape: {esm_embedding_weights.shape}")

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_all_dataloaders(
        processed_data_dir=PROCESSED_DATA_DIR,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    print("\nInitializing model...")
    model = build_model(alphabet_config, esm_embedding_weights)
    count_parameters(model)

    print("\nInitializing Flow Matching components...")
    scheduler = PolynomialConvexScheduler(n=2.0)
    path = MixtureDiscreteProbPath(scheduler=scheduler)
    loss_fn = MixturePathGeneralizedKL(path=path)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5
    )

    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    early_stopping_loss = EarlyStopping(patience=10, min_delta=1e-4, mode='min')

    best_val_loss = float('inf')

    best_loss_ckpt = os.path.join(CHECKPOINT_DIR, 'best_loss_model.ckpt')

    print("\n" + "=" * 60)
    print(f"Starting Training | Loss = FlowKL + {ce_weight} * CE")
    print(f"Saving: best_loss_model.ckpt (based on validation loss)")
    print("=" * 60 + "\n")

    for epoch in range(epochs):
        train_loss, train_flow_loss, train_ce_loss, train_aar, train_perplexity = train_epoch(
            model, train_loader, optimizer, loss_fn, path, alphabet_config, epoch, NOISE_TYPE
        )

        val_loss, val_flow_loss, val_ce_loss, val_aar, val_perplexity = evaluate_epoch(
            model, val_loader, loss_fn, path, alphabet_config, epoch, "Val", NOISE_TYPE
        )

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train - Total: {train_loss:.4f}  Flow: {train_flow_loss:.4f}  "
              f"CE: {train_ce_loss:.4f}  AAR: {train_aar:.4f}  PPL: {train_perplexity:.2f}")
        print(f"  Val   - Total: {val_loss:.4f}  Flow: {val_flow_loss:.4f}  "
              f"CE: {val_ce_loss:.4f}  AAR: {val_aar:.4f}  PPL: {val_perplexity:.2f}")
        print(f"  LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_path=best_loss_ckpt,
                model=model, optimizer=optimizer, scheduler_lr=scheduler_lr,
                epoch=epoch, val_loss=val_loss, val_aar=val_aar,
                val_perplexity=val_perplexity, alphabet_config=alphabet_config,
                save_reason='best_loss'
            )
            print(f"  ✓ [best_loss_model] saved  "
                  f"Val Loss: {val_loss:.4f}  AAR: {val_aar:.4f}  PPL: {val_perplexity:.2f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                save_path=os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.ckpt'),
                model=model, optimizer=optimizer, scheduler_lr=scheduler_lr,
                epoch=epoch, val_loss=val_loss, val_aar=val_aar,
                val_perplexity=val_perplexity, alphabet_config=alphabet_config,
                save_reason=f'periodic_epoch_{epoch + 1}'
            )

        scheduler_lr.step(val_loss)

        stop_loss = early_stopping_loss(val_loss, epoch)
        if stop_loss:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"  Best val loss: {early_stopping_loss.best_score:.4f} "
                  f"at epoch {early_stopping_loss.best_epoch + 1}")
            break

    print("\n" + "=" * 60)
    print("Final Test Set Evaluation...")
    print("=" * 60)

    print(f"\nLoading best_loss checkpoint: {best_loss_ckpt}")
    checkpoint = torch.load(best_loss_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, _, test_aar, _ = evaluate_epoch(
        model, test_loader, loss_fn, path, alphabet_config, 0, "Test", NOISE_TYPE
    )

    print(f"\n[Best Model] Test AAR: {test_aar:.4f}")
    print(f"  Saved at epoch {checkpoint['epoch'] + 1}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  Val AAR: {checkpoint['val_aar']:.4f}")

    print("\n" + "=" * 60)
    print("Training Completed!")
    print(f"  ESM Model:           {alphabet_config.get('model_name', 'esm2_t33_650M_UR50D')}")
    print(f"  Noise Type:          {NOISE_TYPE.upper()}")
    print(f"  Cross Attn Layers:   {cross_attn_layers}")
    print(f"  Cross Attn Heads:    {cross_attn_heads}")
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Test AAR:            {test_aar:.4f}")
    print(f"  Total epochs:        {epoch + 1}")
    print(f"  Checkpoints saved:   {CHECKPOINT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()