from torch.utils.data import DataLoader
from data_preparation import train_dataset, valid_dataset
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)
val_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)