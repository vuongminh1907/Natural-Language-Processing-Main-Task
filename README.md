# 🌐 Natural Language Processing Main Task


This repository focuses on various NLP tasks with support for customizable training configurations.


## 🚀 Run Training


  You can start training your models using two methods:

1. **Direct Command Line Arguments**: 
  Run the training script with flags like `--batch_size`, `--epoch`, etc.
  ```bash
   python train.py --batch_size 16 --epoch 10
  ```
2. **Using a Configuration File**: 

  Create a config file (e.g., `config.txt`) and specify the parameters in each line. Example `config.txt:`

  ```
    batch_size = 19
    model_name = "bert"
    # Add other parameters here...
  ```
  Then run the training script with the config file:
  ```bash
   python train.py --config_file config.txt
  ```

## Text Classification

### Prepare datasets
- Form DatasetDict
```
["attention_mask", "input_ids", "labels", "token_type_ids"]
```
- Creating a file '.txt' with these format
    ```
    'idx' , 'sentence1', 'sentence2', 'label' 
    1,"sentence1_example1","sentence2_example1",1
    ```
- Another way is using HuggingFace datasets

### Running training code

```
python train_text_classify.py
```
### Infer
```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "bert-base-uncased"   #Replacing with your training model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I love coffe", "I hate coffe"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

## Masked Natural Language

### Prepare datasets
- Form DatasetDict
```
['attention_mask', 'input_ids', 'labels', 'word_ids']
```
- Creating a file '.txt' with these format
    ```
    'idx' , 'sentence' 
    1,"sentence1_example1"
    ```
- Another way is using HuggingFace datasets

### Running training code

```
python train_masked_NL.py
```
### Infer
```
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"  #Replacing with your model path 
)
text = "I love the [MASK]"
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")
```

