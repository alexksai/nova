import requests
import os
import json
from dotenv import load_dotenv
import bittensor as bt
from datasets import load_dataset

import asyncio

load_dotenv(override=True)

def upload_file_to_github(filename: str, encoded_content: str):
    # Github configs
    github_repo_name = os.environ.get('GITHUB_REPO_NAME')   # example: nova
    github_repo_branch = os.environ.get('GITHUB_REPO_BRANCH') # example: main
    github_token = os.environ.get('GITHUB_TOKEN')
    github_repo_owner = os.environ.get('GITHUB_REPO_OWNER') # example: metanova-labs
    github_repo_path = os.environ.get('GITHUB_REPO_PATH') # example: /data/results or ""

    if not github_repo_name or not github_repo_branch or not github_token or not github_repo_owner:
        raise ValueError("Github environment variables not set. Please set them in your .env file.")

    target_file_path = os.path.join(github_repo_path, f'{filename}.txt')
    url = f"https://api.github.com/repos/{github_repo_owner}/{github_repo_name}/contents/{target_file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        }

    # Check if the file already exists (need its SHA to update)
    existing_file = requests.get(url, headers=headers, params={"ref": github_repo_branch})
    sha = existing_file.json().get("sha") if existing_file.status_code == 200 else None

    payload = {
        "message": f"Encrypted response for {filename}",
        "content": encoded_content,
        "branch": github_repo_branch,
    }
    if sha:
        payload["sha"] = sha  # updating existing file

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        return True
    else:
        bt.logging.error(f"Failed to upload file for {filename}: {response.status_code} {response.text}")
        return False


def get_smiles(product_name):
    # Remove single and double quotes from product_name if they exist
    if product_name:
        product_name = product_name.replace("'", "").replace('"', "")
    else:
        bt.logging.error("Product name is empty.")
        return None

    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")

def get_random_protein():
    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set contact nova team for api key.")

    url = "https://rvhs77j663.execute-api.us-east-1.amazonaws.com/prod/random-protein-of-interest"
    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} {response.text}")

    data = response.json()

    if "body" in data:
        inner_body_str = data["body"]  # e.g. '{"uniprot_code": "A0S183", "protein_sequence": "..."}'
        try:
            inner_data = json.loads(inner_body_str) 
            return inner_data.get("uniprot_code")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse body as JSON: {e}")
    else:
        bt.logging.error("Unexpected API response structure.")

def get_sequence_from_protein_code(protein_code:str) -> str:

    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code != 200:
        return None
    else:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        return amino_acid_sequence

def get_index_in_range_from_blockhash(block_hash: str, range_max: int) -> int:

    block_hash_str = block_hash.lower().removeprefix('0x')
    
    # Convert the hex string to an integer
    hash_int = int(block_hash_str, 16)

    # Modulo by the desired range
    random_index = hash_int % range_max

    return random_index

def get_protein_code_at_index(index: int) -> str:
    
    dataset = load_dataset("Metanova/Proteins", split="train")
    row = dataset[index]  # 0-based indexing
    return row["Entry"]
