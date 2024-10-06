import argparse

def parse_config_file(file_path):
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Loại bỏ dòng trống hoặc dòng comment
            line = line.strip()
            if line and not line.startswith('#'):
                # Tách thành key và value
                key, value = line.split('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")  # Loại bỏ dấu ngoặc kép nếu có
                
                # Kiểm tra và chuyển đổi giá trị kiểu int nếu cần
                if value.isdigit():
                    value = int(value)
                config[key] = value
    return config

def get_params():
    parser = argparse.ArgumentParser(description='Get the parameters for training')

    # Dataset and model parameters
    parser.add_argument('--dataset_path', type=str, default=None, help="Path to custom dataset file.")
    parser.add_argument('--checkpoint', type=str, default="bert-base-uncased", help="Pretrained model checkpoint.")
    parser.add_argument('--model_name', type=str, default="None", help="Name of the model to be trained.")
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate for optimizer.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs.")
    parser.add_argument('--vocab_size', type=int, default=None, help="Vocabulary size for training tokenizer.")

    # Hugging Face API parameters
    parser.add_argument('--hf_token', type=str, required=False, help="Hugging Face API token.")
    parser.add_argument('--repo_id', type=str, required=False, help="Hugging Face repository ID.")

    parser.add_argument('--config_file', type=str, default=None, help='Path to the configuration file')

    args = parser.parse_args()

    if args.config_file:
        config = parse_config_file(args.config_file)
        for key, value in config.items():
            # Nếu tham số đã có trong argparse, ghi đè nó
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                # Nếu tham số chưa có trong argparse, thêm nó vào args
                setattr(args, key, value)

    return args

