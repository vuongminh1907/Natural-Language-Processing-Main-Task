import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

import torch
from tqdm.auto import tqdm
from utils.get_dataset import get_custom_dataset
from utils.get_params import get_params
from utils.upload_hug import upload_hug

def group_texts(examples,chunk_size=128):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length < chunk_size:
        chunk_size = total_length//2
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

def tokenize_function_mask(examples, tokenizer):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

if __name__ == "__main__":
    #get the parameters for training
    args = get_params()

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #Load model and tokenizer
    if args.checkpoint is not None:
        model_checkpoint = args.checkpoint
    else:
        model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

   
    # Load the dataset
    if args.dataset_path is not None:
        imdb_dataset = get_custom_dataset(args.dataset_path)
    else:
        imdb_dataset = load_dataset("imdb")

 
    tokenized_datasets = imdb_dataset.map(
        lambda examples: tokenize_function_mask(examples, tokenizer),
        batched=True, remove_columns=["text", "label"]
    )

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    if args.dataset_path is None:
        #down sample the dataset
        train_size = 10_000
        test_size = int(0.1 * train_size)

        downsampled_dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
    else:
        lm_datasets.pop('validation')
        downsampled_dataset = lm_datasets

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

    batch_size = args.batch_size if args.batch_size is not None else 64
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_train_epochs = args.num_epochs if args.num_epochs is not None else 3   
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(args.num_epochs):
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

    #Upoad huggingface model
    if args.repo_id is not None:
        upload_hug(args,model=model,tokenizer=tokenizer)
    

        



