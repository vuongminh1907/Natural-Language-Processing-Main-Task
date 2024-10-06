from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForQuestionAnswering , AdamW
from transformers import get_scheduler, default_data_collator
from datasets import load_dataset

from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.get_dataset import get_custom_dataset
from utils.get_params import get_params
from utils.upload_hug import upload_hug



if __name__ == "__main__":
    #get the parameters for training
    args = get_params()
    #Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if args.dataset_path is not None:
        raw_datasets = get_custom_dataset(args.dataset_path)
    else:
        raw_datasets = load_dataset("squad")

    model_checkpoint = args.checkpoint if args.checkpoint else "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)

    max_length = 384        # The maximum length of a feature (question and context)
    stride = 128            # The authorized overlap between two part of the context when splitting it is needed.

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # def preprocess_validation_examples(examples):
    #     questions = [q.strip() for q in examples["question"]]
    #     inputs = tokenizer(
    #         questions,
    #         examples["context"],
    #         max_length=max_length,
    #         truncation="only_second",
    #         stride=stride,
    #         return_overflowing_tokens=True,
    #         return_offsets_mapping=True,
    #         padding="max_length",
    #     )

    #     sample_map = inputs.pop("overflow_to_sample_mapping")
    #     example_ids = []

    #     for i in range(len(inputs["input_ids"])):
    #         sample_idx = sample_map[i]
    #         example_ids.append(examples["id"][sample_idx])

    #         sequence_ids = inputs.sequence_ids(i)
    #         offset = inputs["offset_mapping"][i]
    #         inputs["offset_mapping"][i] = [
    #             o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
    #         ]

    #     inputs["example_id"] = example_ids
    #     return inputs

    # validation_dataset = raw_datasets["validation"].map(
    #     preprocess_validation_examples,
    #     batched=True,
    #     remove_columns=raw_datasets["validation"].column_names,
    # )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8,
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    #define the parameters
    num_epochs = args.num_epochs
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


    #upload the model to the hub
    if args.repo_id is not None:
        upload_hug(model, tokenizer, args.repo_id)



