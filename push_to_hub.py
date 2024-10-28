from huggingface_hub import Repository
import shutil
import os

# token
token = os.environ['HF_TOKEN']

# repo paths
repo_id = "azizbp/trashnet-densenet121"
local_repo_dir = "hf_repo"

# clone the hugging face repo
repo = Repository(local_dir=local_repo_dir, clone_from=repo_id, use_auth_token=token)

# copy model file
model_file = "bestModel-trashnet_v9-densenet121.h5"

if os.path.exists(model_file):
    shutil.copy(model_file, local_repo_dir)
else:
    raise FileNotFoundError(f"Model file {model_file} not found")

# push to hub
repo.push_to_hub(commit_message="Automated model push from CI/CD pipeline")