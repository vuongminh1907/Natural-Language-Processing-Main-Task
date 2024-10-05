# 🌐 Natural Language Processing Main Task


This repository focuses on various NLP tasks with support for customizable training configurations.


## 🚀 Run Training

  You can start training your models using two methods:

1. **Direct Command Line Arguments**: 
  
2. **Using a Configuration File**: 

  Read through [config.md](https://github.com/vuongminh1907/Natural-Language-Processing-Main-Task/blob/main/docs/config.md) for instruction.

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

## 🌍 Translation
This section covers how to train a translation model that converts text from one language to another.

### 📦 Dataset Preparation

  Ensure your dataset follows the `DatasetDict` format with the following features:
  ```
  DatasetDict({
            train: Dataset({
                features: ['id', 'translation'],
                num_rows: 210173
            })
        })
  ```

To create the dataset:
- Save a `translation_dataset.txt` file with this format:
```
idx,en,vie
1,"hello","Xin chào"
```
- Alternatively, you can use the **Hugging Face Datasets** library to load and process datasets directly.

### 🏃‍♂️ Run Training
To train the translation model:
```
python train_trans.py
```
### 🔍 Inference
```
python infer_trans.py
```

## 📝 Summarization
This section covers how to train a summarization model that converts long text into a concise summary.

### 📦 Dataset Preparation

  Ensure your dataset follows the `DatasetDict` format with the following features:
  ```
  DatasetDict({
            train: Dataset({
                features: ['Unnamed: 0', 'Document', 'Summary'],
                num_rows: ....
            })
        })
  ```

To create the dataset:
- Save a `summarization_dataset.txt` file with this format:
```
idx,Document,Summary
1,"Sáng nay, tại Hà Nội, một cuộc họp quan trọng đã diễn ra giữa các nhà lãnh đạo doanh nghiệp lớn trong nước. Cuộc họp tập trung vào các vấn đề phát triển kinh tế sau đại dịch và những thách thức mà các doanh nghiệp phải đối mặt. Các đại biểu đã thảo luận về việc tăng cường hợp tác công tư, đổi mới công nghệ, và cải thiện năng suất lao động để thúc đẩy tăng trưởng kinh tế. Các chuyên gia kinh tế cũng nhấn mạnh tầm quan trọng của việc đào tạo nguồn nhân lực chất lượng cao và áp dụng các biện pháp hỗ trợ doanh nghiệp vượt qua khó khăn.","Cuộc họp tại Hà Nội bàn về phát triển kinh tế sau đại dịch và thảo luận các biện pháp hỗ trợ doanh nghiệp."

```
- Alternatively, you can use the **Hugging Face Datasets** library to load and process datasets directly.

### 🏃‍♂️ Run Training
To train the translation model:
```
python train_summarization.py
```
### 🔍 Inference
```
python infer_sum.py
```


## 🛠️ Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)



