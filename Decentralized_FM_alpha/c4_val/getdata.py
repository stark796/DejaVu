from datasets import load_dataset
from tqdm import tqdm
import json


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


# Exact replication of OPT data collection setup
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
dataset = dataset.shuffle(buffer_size=10000, seed=42)
path = "c4_valid.jsonl"

for idx, doc in enumerate(tqdm(dataset)):
    # Truncate very long documents to avoid sequence length issues
    # (max ~8000 chars to stay under 2048 tokens with typical tokenization)
    text = doc["text"][:8000] if len(doc["text"]) > 8000 else doc["text"]
    
    data = {
        "best_of": 1,
        "echo": True,
        "logprobs": 1,
        "max_tokens": 0,
        "model": "opt-175b",
        "n": 1,
        "prompt": text,
        "request_type": "language-model-inference",
        "stop": None,
        "temperature": 0,
        "top_p": 1,
    }
    dump_jsonl([data], path, append=True)
    if idx == 500:
        print("Collected 500 samples")
        break
