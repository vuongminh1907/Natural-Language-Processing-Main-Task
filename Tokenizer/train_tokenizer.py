from datasets import load_dataset
from transformers import AutoTokenizer
from utils.get_params import get_params

def get_training_corpus(dataset,args):
    if args.dataset_name is not None:
        return (
            dataset  for i in range(0, len(dataset["train"]), 1000)
        )
    else:
        return (
        dataset["train"][i : i + 1000]["original"]
        for i in range(0, len(dataset["train"]), 1000)
    )

if __name__ == "__main__":
    
    #get parameters
    args = get_params()

    #load the dataset

    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name)
    else:
        #Load the vietnamese dataset
        dataset = load_dataset("thanhkt/vietnam-normalize-24k")
    print("Loaded the dataset successfully")

    #load the tokenizer
    if args.tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        #Load the vietnamese tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    #get the training corpus
    training_corpus = get_training_corpus(dataset,args)

    #train the tokenizer
    new_tokenizer = tokenizer.train_from_iterator(training_corpus,args.vocab_size if args.vocab_size is not None else 52000)
    print("Training the tokenizer successfully")

    #save the tokenizer
    new_tokenizer.push_to_hub(args.tokenizer_name if args.tokenizer_name is not None else "vietnamese-tokenizer")

    