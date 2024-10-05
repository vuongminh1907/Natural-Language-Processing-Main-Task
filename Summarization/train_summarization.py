from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamW
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from utils.get_params import get_params
from utils.get_dataset import get_custom_dataset
from utils.upload_hug import upload_hug

if __name__ == "__main__":
    # get the parameters for training
    args = get_params()

    if args.dataset_path is not None:
        dataset = get_custom_dataset(args.dataset_path)
    else:
        #load vietnamese summarization dataset
        dataset = load_dataset("OpenHust/vietnamese-summarization")

    #define model checkpoint
    if args.checkpoint is not None:
        checkpoint = args.model_name
    else:
        checkpoint = "google/mt5-small"
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    max_input_length = 512
    max_target_length = 30

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["Document"],
            max_length=max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            examples["Summary"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    """
        DatasetDict({
            train: Dataset({
                features: ['Unnamed: 0', 'Document', 'Summary', 'Dataset', 'input_ids', 'attention_mask', 'labels'],
                num_rows: 74564
            })
        })
    """

    #remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset["train"].column_names
    )
    """
    DatasetDict({
        train: Dataset({
            features: ['input_ids', 'attention_mask', 'labels'],
            num_rows: 74564
        })
    })
    """
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    #define the dataloader
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer= optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    #start training
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            progress_bar.update(1)
            

    #upload hugging face
    if args.repo_id is not None:
        upload_hug(args,model=model,tokenizer=tokenizer)





    