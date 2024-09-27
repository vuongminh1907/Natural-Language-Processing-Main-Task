from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split


def get_custom_dataset(file_path):
     
    with open(file_path, 'r') as file:
        column_names = file.readline().strip().replace("'", "").split(", ")
    column_names = [name.strip() for name in column_names] 

    data = pd.read_csv(file_path, header=None, skiprows=1, names=column_names, quotechar='"')

    train_data, temp_data = train_test_split(data, test_size=0.8, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    validation_dataset = Dataset.from_pandas(validation_data, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_data, preserve_index=False)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })

    return dataset_dict