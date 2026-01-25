

import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_att import train

# Data paths for Llama models
DATA = {
    "llama-3b": {
        "c4": "../Decentralized_FM_alpha/data/llama_3b_c4",
    },
    "llama-8b": {
        "c4": "../Decentralized_FM_alpha/data/llama_8b_c4",
    },
}

MODEL_CHOICES = ['llama-3b', 'llama-8b']
DATA_CHOICES = ['c4']

# Configuration for Llama models
CONFIG = {
    'llama-3b': {
        'num_layer': 28,
        'ckt_storage': "bylayer",
        'd': 3072,
        'intermediate': 8192,
        'h': 24,
        'kv_h': 8,
        'N': 80000,
    },
    'llama-8b': {
        'num_layer': 32,
        'ckt_storage': "bylayer",
        'd': 4096,
        'intermediate': 14336,
        'h': 32,
        'kv_h': 8,
        'N': 80000,
    },
}


class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train):
        self.X = X
        self.Y = Y
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        
        # Check for degenerate case
        if y.sum() == 0:
            print("Warning: all zero y encountered")
        return x, y


def get_data(args, layer_idx):
    """
    Load training data for attention predictor.
    
    Note: For attention prediction, we use the input to the CURRENT layer
    to predict which attention heads are important in that layer.
    This is different from OPT which uses layer L-1 input for layer L prediction.
    """
    config = CONFIG[args.model]
    
    if config['ckt_storage'] == "bylayer":
        # Load attention input (query)
        # Use layer_idx-1 for lookahead prediction (like OPT)
        query_layer = max(0, layer_idx - 1)
        path = f"{DATA[args.model][args.dataset]}/att_x_{query_layer}.mmap"
        print(f"Reading query from {path}")
        query = np.array(np.memmap(
            path, dtype='float16', mode='r',
            shape=(config['N'], config['d'])
        ))
        
        # Load attention head importance label (output norm per head)
        path = f"{DATA[args.model][args.dataset]}/att_label_{layer_idx}.mmap"
        print(f"Reading attention label from {path}")
        label = np.array(np.memmap(
            path, dtype='float16', mode='r',
            shape=(config['N'], config['h'])
        ))
        
        # Filter out samples where label sum is 0 (invalid)
        num_valid = (label.sum(-1) > 0).sum()
        print(f"Valid samples: {num_valid}/{config['N']}")
        
        return query[:num_valid], label[:num_valid]


def create_dataset(query, labels, args):
    """Create training and test dataloaders."""
    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)
    
    print(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    print(f"# training data: {num_train}, # test data: {num_test}")
    
    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)
    
    train_dataloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    
    return train_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description="Llama Attention Sparse Predictor Training")
    parser.add_argument("--model", type=str, default="llama-3b", choices=MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="c4", choices=DATA_CHOICES)
    parser.add_argument("--L", type=int, default=0, help="which layer")
    parser.add_argument("--D", type=int, default=1000, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--k", type=float, default=0.7, 
                       help="top k percent to mark as active head")
    args = parser.parse_args()
    
    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = CONFIG[args.model]
    
    print("=" * 40, f"Layer {args.L}", "=" * 40)
    print(f"Model: {args.model}")
    print(f"Hidden size: {config['d']}")
    print(f"Num attention heads: {config['h']}")
    print(f"Num KV heads (GQA): {config['kv_h']}")
    
    query, labels = get_data(args, args.L)
    train_loader, test_loader = create_dataset(query, labels, args)
    
    # Predictor: maps hidden_size → num_attention_heads
    predictor = torch.nn.Sequential(
        torch.nn.Linear(config['d'], args.D, bias=None),
        torch.nn.Linear(args.D, config['h'], bias=None),
    )
    
    print(f"\nPredictor architecture:")
    print(f"  Input: {config['d']} (hidden_size)")
    print(f"  Hidden: {args.D} (low rank)")
    print(f"  Output: {config['h']} (num_attention_heads)")
    
    print("\nStart Training")
    best_model, eval_result = train(
        predictor, train_loader, test_loader, args, device, verbal=True
    )
    
    # Save checkpoint
    import os
    checkpoint_dir = f"../checkpoint/llama-{args.model.replace('llama-', '')}-sparse-predictor"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    path = f"{checkpoint_dir}/{args.dataset}_att_k{args.k}_layer{args.L}_recall-{eval_result['Recall']:.4f}-sparsity-{eval_result['Classifier Sparsity']:.0f}.pt"
    torch.save(best_model, path)
    print(f"\nSaved checkpoint to: {path}")
    
    # Print final results for comparison with OPT
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Compare with OPT to verify DejaVu hypothesis)")
    print("=" * 60)
    print(f"Layer {args.L}:")
    print(f"  Recall: {eval_result['Recall']:.4f}")
    print(f"  True Sparsity: {eval_result['True Sparsity']:.2f}")
    print(f"  Classifier Sparsity: {eval_result['Classifier Sparsity']:.2f}")
    print(f"  Loss: {eval_result['Loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
