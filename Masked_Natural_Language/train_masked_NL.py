from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import get_scheduler, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

import torch
import argparse
import math
from tqdm.auto import tqdm

def group_texts(examples,chunk_size=128):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    # Argument parsing for command-line options
    parser = argparse.ArgumentParser(description='Train a BERT model for masked language modeling')
    parser.add_argument('--num_epoch', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the configuration file')

    args = parser.parse_args()

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #Load model and tokenizer
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load the dataset
    imdb_dataset = load_dataset("imdb")

    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    
    tokenized_datasets = imdb_dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "label"]
    )

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    #down sample the dataset
    train_size = 10_000
    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}
    
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    eval_dataset = downsampled_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

    batch_size = 64
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.item())
        print(f"Epoch {epoch} - Evaluation loss: {sum(losses) / len(losses)}")

    # Save the model
    model.save_pretrained("masked_NL_model")
    tokenizer.save_pretrained("masked_NL_model")
    print("Model saved")
    

        


