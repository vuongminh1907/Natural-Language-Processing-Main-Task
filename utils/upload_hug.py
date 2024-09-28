from huggingface_hub import HfApi, HfFolder

def upload_hug(args, model=None, tokenizer=None):
    """
    Upload a model to the Hugging Face Hub
    """
    api = HfApi()

    #Check if the repo already exists
    try:
        api.repo_info(args.repo_id, use_auth_token=args.hf_token)
        print(f"Repository '{args.repo_id}' already exists.")
    except Exception as e:
        print("Creating a new repository")
        api.create_repo(args.repo_id, token=args.hf_token, private=False)
    
    #push the model
    if model is not None:
        model.push_to_hub(repo_id=args.repo_id, token=args.hf_token)
        print("Model uploaded")
    
    #push the tokenizer
    if tokenizer is not None:
        tokenizer.push_to_hub(repo_id=args.repo_id, token=args.hf_token)
        print("Tokenizer uploaded")