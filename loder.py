import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class PeptideDataset(Dataset):
    def __init__(self, processed_data_path):
        with open(processed_data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.length = len(self.data['binder_input_ids'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'binder_input_ids': self.data['binder_input_ids'][idx],
            'binder_attention_mask': self.data['binder_attention_mask'][idx],
            'binder_lengths': self.data['binder_lengths'][idx],
            'target_embeddings': self.data['target_embeddings'][idx],
            'target_attention_mask': self.data['target_attention_mask'][idx],
            'target_lengths': self.data['target_lengths'][idx],
            'raw_binder': self.data['raw_binder'][idx],
            'raw_target': self.data['raw_target'][idx]
        }


def collate_fn(batch):
    binder_input_ids = torch.stack([item['binder_input_ids'] for item in batch])
    binder_attention_mask = torch.stack([item['binder_attention_mask'] for item in batch])
    binder_lengths = torch.stack([item['binder_lengths'] for item in batch])
    target_embeddings = torch.stack([item['target_embeddings'] for item in batch])
    target_attention_mask = torch.stack([item['target_attention_mask'] for item in batch])
    target_lengths = torch.stack([item['target_lengths'] for item in batch])
    raw_binder = [item['raw_binder'] for item in batch]
    raw_target = [item['raw_target'] for item in batch]

    return {
        'binder_input_ids': binder_input_ids,
        'binder_attention_mask': binder_attention_mask,
        'binder_lengths': binder_lengths,
        'target_embeddings': target_embeddings,
        'target_attention_mask': target_attention_mask,
        'target_lengths': target_lengths,
        'raw_binder': raw_binder,
        'raw_target': raw_target
    }


def get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4):
    dataset = PeptideDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader


def get_all_dataloaders(processed_data_dir, batch_size=32, num_workers=4):
    train_loader = get_dataloader(
        os.path.join(processed_data_dir, 'train.pkl'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = get_dataloader(
        os.path.join(processed_data_dir, 'val.pkl'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = get_dataloader(
        os.path.join(processed_data_dir, 'test.pkl'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader