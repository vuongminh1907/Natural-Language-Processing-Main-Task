from utils.get_params import get_params

args = get_params()
 # Sử dụng các tham số trong logic của bạn
print(f"Batch size: {args.batch_size}")
print(f"Number of epochs: {args.num_epoch}")
print(f"Model name: {args.model_name}")

# In ra các tham số từ file config không có trong argparse
print("Other parameters from config_file:")
for key in vars(args):
    if key not in ['batch_size', 'num_epoch', 'data_dir', 'model_name', 'config_file']:
        print(f"{key}: {getattr(args, key)}")

