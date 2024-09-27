from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification, AdamW
from transformers import get_scheduler
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import evaluate
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from huggingface_hub import HfApi
from utils.get_dataset import get_custom_dataset


if __name__ == "__main__":

    # Argument parsing for command-line options
    parser = argparse.ArgumentParser(description='Train a BERT model for sequence classification')
    parser.add_argument('--num_epoch', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the configuration file')

    args = parser.parse_args()

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    path = "sample.txt"
    path = None
    # Load the dataset
    if path is not None:
        raw_datasets = get_custom_dataset(path)
    else:
        raw_datasets = load_dataset("glue", "mrpc")

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    #get the columns
    columns = tokenized_datasets["train"].column_names
    for col in columns:
        if col not in ['labels', 'input_ids', 'token_type_ids', 'attention_mask']:
            tokenized_datasets = tokenized_datasets.remove_columns(col)
    tokenized_datasets.set_format("torch")
    print("Columns: ", tokenized_datasets["train"].column_names)
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments("test_trainer")
    
    #define the dataloader
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    #define the parameters
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    #start training
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

   
    #evaluate the model
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        eval_loss += outputs.loss.item()
        eval_steps += 1
    eval_loss = eval_loss / eval_steps
    print("Eval loss: ", eval_loss)

    #model name
    model_name = "test_trainer"
    #save the model
    model.save_pretrained(model_name)
    #save the tokenizer
    tokenizer.save_pretrained(model_name)
    print("Model saved")

    api = HfApi()
    api.upload_folder(
        folder_path=model_name, 
        repo_id="your_huggingface_username/your_model_repo", 
        repo_type="model",
        token="Your tokens" # Truyền token ở đây
    )
    print("Model uploaded")

