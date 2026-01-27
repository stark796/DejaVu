#!/usr/bin/env python3
"""Download WikiText-2 and prepare it for perplexity evaluation."""

import os
import json
from datasets import load_dataset

def wikitext_detokenize(string):
    """Clean up WikiText formatting."""
    import re
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def main():
    print("Downloading WikiText-2 dataset...")
    
    # Load WikiText-2 (raw version for proper detokenization)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Create output directories
    os.makedirs("./data/wikitext", exist_ok=True)
    os.makedirs("./wikitext_eval", exist_ok=True)
    
    # Save splits for potential other uses
    print("Saving train split to disk...")
    dataset["train"].save_to_disk("./data/wikitext/train")
    print("Saving test split to disk...")
    dataset["test"].save_to_disk("./data/wikitext/test")
    print("Saving validation split to disk...")
    dataset["validation"].save_to_disk("./data/wikitext/validation")
    
    # Create JSONL file for inference
    # Combine all test text into chunks of ~2000 chars for context
    print("\nPreparing JSONL file for inference...")
    
    # Get all non-empty text from test set
    texts = [wikitext_detokenize(t) for t in dataset["test"]["text"] if t.strip()]
    full_text = "\n\n".join(texts)
    
    # Split into chunks for inference (each chunk becomes a prompt)
    chunk_size = 1500  # characters per prompt
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size]
        if len(chunk.strip()) > 100:  # Skip very short chunks
            chunks.append(chunk)
    
    # Write JSONL file
    jsonl_path = "./wikitext_eval/wikitext_test.jsonl"
    with open(jsonl_path, "w") as f:
        for chunk in chunks:
            request = {
                "prompt": chunk,
                "echo": True,
                "max_tokens": 0,  # Just compute logprobs, don't generate
            }
            f.write(json.dumps(request) + "\n")
    
    print(f"\nDataset saved:")
    print(f"  Raw splits: ./data/wikitext/")
    print(f"  JSONL for inference: {jsonl_path}")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Total test samples: {len(dataset['test'])}")

if __name__ == "__main__":
    main()
