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

## 🔤 Tokenizer Training
In this section, you'll learn how to train a tokenizer from maybe one language to another language.
### 📦 Dataset Preparation
Example format of `corpus.txt`:
```
Tôi yêu Việt Nam.
Cho tôi một like nhé, yêu.
Đây là thử nghiệm NLP.
```
### 🏃‍♂️ Run Training
To train a tokenizer:
```
python train_tokenizer.py
```
## 🧑‍💻 Text Classification
Text Classification is a fundamental NLP task where the goal is to categorize text into predefined labels.


### 📦 Dataset Preparation

  Ensure your dataset follows the `DatasetDict` format with the following features:
  ```
  ["attention_mask", "input_ids", "labels", "token_type_ids"]
  ```

To create the dataset:
- Save a `.txt` file with this format:
```
idx,sentence1,sentence2,label
1,"Example sentence 1","Example sentence 2",1
```
- Alternatively, you can use the **Hugging Face Datasets** library to load and process datasets directly.

### 🏃‍♂️ Run Training
To train the text classification model:
```
python train_text_classify.py
```
### 🔍 Inference
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

## 📝 Masked Natural Language Modeling
Masked Language Modeling involves predicting missing words in a sentence, allowing models to learn the context and relationships between words.


### 📦 Dataset Preparation

  Ensure your dataset follows the `DatasetDict` format with the following features:
  ```
  ["attention_mask", "input_ids", "labels", "token_type_ids"]
  ```

To create the dataset:
- Save a `.txt` file with this format:
```
idx,sentence
1,"Example sentence 1"
```
- Alternatively, you can use the **Hugging Face Datasets** library to load and process datasets directly.

### 🏃‍♂️ Run Training
To train the masked language model:
```
python train_masked_NL.py
```
### 🔍 Inference
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

## 🛠️ Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)



