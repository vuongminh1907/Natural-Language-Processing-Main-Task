
from transformers import get_scheduler, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

import torch
from tqdm.auto import tqdm
from utils.get_dataset import get_custom_dataset
from utils.get_params import get_params


if __name__ == "__main__":
    #get the parameters for training
    args = get_params()

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    