import torch
import numpy as np
import h5py
from utils import device


class FontDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: list, letter_idx: int, chunk_size: int, chunk_idx: int) -> None:
        self.letter_idx = letter_idx
        self.chunk_size = chunk_size
        self.chunk_idx = chunk_idx

        with h5py.File(dataset_path, "r") as f:
            entire_dataset = f.get("fonts")
            self.chunk_idx = np.clip(chunk_idx, 0, (len(entire_dataset) // self.chunk_size))
            start_idx = self.chunk_size*self.chunk_idx
            end_idx = np.clip(self.chunk_size*(self.chunk_idx + 1), 0, 56442)

            self.dataset = entire_dataset[start_idx:end_idx]

    def __getitem__(self, font_idx: int) -> torch.Tensor:
        sample = self.dataset[font_idx][self.letter_idx]
        return torch.tensor(sample).unsqueeze(0).to(device=device) / 255.0

    def __len__(self):
        return self.chunk_size


def create_dataloader(dataset_path: str, letter_idx: int, chunk_size: int, chunk_idx: int, batch_size: int):
    fontDataset = FontDataset(dataset_path, letter_idx, chunk_size, chunk_idx)
    return torch.utils.data.DataLoader(
        fontDataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
