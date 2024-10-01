
from transformers import get_scheduler, default_data_collator, pipeline, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

import torch
from tqdm.auto import tqdm
from utils.get_dataset import get_custom_dataset
from utils.get_params import get_params
from utils.upload_hug import upload_hug
import numpy as np


if __name__ == "__main__":
    #get the parameters for training
    args = get_params()

    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    #Load the dataset
    if args.dataset_path:
        dataset = get_custom_dataset(args.dataset_path)
    else:
        dataset = load_dataset("kde4", lang1="en", lang2="fr")
        """
        DatasetDict({
            train: Dataset({
                features: ['id', 'translation'],
                num_rows: 210173
            })
        })
        dataset["train"][1]["translation"]
        Example:
        {'en': 'Default to expanded threads',
        'fr': 'Par défaut, développer les fils de discussion'}
        """

        #Split the dataset
        dataset = dataset.train_test_split(train_size=0.8, seed=42)
        #Rename test to validation
        dataset['validation'] = dataset.pop('test')
        
    #Load model

    if args.checkpoint:
        model_checkpoint = args.checkpoint
    else:
        model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    #Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    """
    Example for processing the dataset
    en_sentence = split_datasets["train"][1]["translation"]["en"]
    fr_sentence = split_datasets["train"][1]["translation"]["fr"]

    inputs = tokenizer(en_sentence, text_target=fr_sentence)
    inputs

    {'input_ids': [47591, 12, 9842, 19634, 9, 0], 'attention_mask': [1, 1, 1, 1, 1, 1], 'labels': [577, 5891, 2, 3184, 16, 2542, 5, 1710, 0]}
    """
    

    max_length = 128
    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        # The reason why need to return the labels is that If you forget to indicate that you are tokenizing labels, they will be tokenized by the input tokenizer
        return model_inputs
    
    #Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True,remove_columns=dataset["train"].column_names)

    #Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    tokenized_datasets.set_format("torch")

    #Define the dataloader
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator)


    #Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    #Scheduler
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)\
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    def postprocess(predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels
    
    #Training

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        #Evaluation
        model.eval()
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

        #Postprocess
        predictions = logits.argmax(dim=-1)
        decoded_preds, decoded_labels = postprocess(predictions, batch["labels"])

        print(f"Epoch {epoch} Loss: {loss}")
        print(f"Predictions: {decoded_preds}")
        print(f"Labels: {decoded_labels}")

    #upload hugging face
    if args.repo_id is not None:
        upload_hug(args,model=model,tokenizer=tokenizer)



