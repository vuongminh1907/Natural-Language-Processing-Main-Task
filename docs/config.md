# 🌐 Parameter Setup for Training Script 

This guide will walk you through how to configure and run the training script using either direct command-line arguments or a configuration file.

## 📜 Script Overview
 This script accepts various parameters for training a model, such as dataset path, model name, learning rate, batch size, and Hugging Face API token. You can pass these parameters either: 
 1. **Directly via command-line arguments**. 
 2. **By providing a configuration file** with parameters.

 ## 🏃‍♂️ Running the Script
1. Direct Command-Line Arguments You can pass parameters directly when running the script from the command line using flags. 

For example:
```
python train.py --dataset_path "/data/my_dataset.csv" --model_name "bert" --learning_rate 3e-5 --batch_size 16 --num_epochs 5 --hf_token "your_token" --repo_id "your_repo"
```

## 📜 Command-Line Arguments

The following table lists the command-line arguments you can use to configure your training script:

| Argument        | Type   | Description                                          | Default              | Required |
|-----------------|--------|------------------------------------------------------|----------------------|----------|
| `--dataset_path`| string | Path to the custom dataset file                       | `None`               | No       |
| `--checkpoint`  | string | Pretrained model checkpoint                          | `"bert-base-uncased"`| No       |
| `--model_name`  | string | Name of the model to be trained                      | `"None"`             | No       |
| `--learning_rate`| float | Learning rate for optimizer                          | `5e-5`               | No       |
| `--batch_size`  | int    | Batch size for DataLoader                            | `8`                  | No       |
| `--num_epochs`  | int    | Number of training epochs                            | `3`                  | No       |
| `--vocab_size`  | int    | Vocabulary size for training tokenizer               | `None`               | No       |
| `--hf_token`    | string | Hugging Face API token                               | None                 | Yes      |
| `--repo_id`     | string | Hugging Face repository ID                           | None                 | Yes      |
| `--config_file` | string | Path to a configuration file                         | `None`               | No       |

 2. Using a Configuration File 
 
 You can also set the parameters in a configuration file and pass it to the script using the `--config_file` argument. 
 
 Example of `config.txt`:

 ```
 dataset_path = "/data/my_dataset.csv"
model_name = "bert"
learning_rate = 3e-5
batch_size = 16
num_epochs = 5
hf_token = "your_token"
repo_id = "your_repo"
```

Running the Script with the Configuration File
```
python train.py --config_file config.txt
```

## 🛠️ Additional Notes
The configuration file allows for flexibility in managing multiple training setups.

You can mix command-line arguments and the config file. For example, if you want to override a specific parameter from the config file, just pass it via the command line:

```
python train.py --config_file config.txt --learning_rate 2e-5
```
This will use the parameters from `config.txt` but override the `learning rate` to `2e-5`.
