#!/usr/bin/env python3
"""
Debug script to verify model predictions match ground truth.
"""
import json
import os
from collections import defaultdict

def main():
    # Load the output results
    results_file = "mmlu_business_ethics_output.jsonl"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    # Load results
    results = {}
    with open(results_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Key by prompt
                prompt = item["request"]["prompt"]
                result = item["result"]
                if "choices" in result:  # loglikelihood
                    logprobs = result["choices"][0]["logprobs"]
                    # Get last token's logprob
                    last_logprob = logprobs["token_logprobs"][-1]
                    last_token = logprobs["tokens"][-1]
                    
                    # Group by context (everything except last "Answer: X")
                    # The prompt ends with "Answer: A" or "Answer: B" etc
                    parts = prompt.rsplit("Answer:", 1)
                    if len(parts) == 2:
                        ctx = parts[0] + "Answer:"
                        answer_letter = parts[1].strip()
                        
                        if ctx not in results:
                            results[ctx] = {}
                        results[ctx][answer_letter] = {
                            "logprob": last_logprob,
                            "token": last_token
                        }
    
    print(f"Loaded {len(results)} unique questions with {sum(len(v) for v in results.values())} total responses")
    
    # Now check the first 10 questions
    questions = sorted(results.keys())[:10]
    
    # Load ground truth from lm_eval (we'll just verify manually based on MMLU)
    # For MMLU business_ethics, the answers are from the cais/mmlu dataset
    
    # Let's just show what the model picked
    print("\n" + "="*60)
    print("MODEL PREDICTIONS (first 10 questions)")
    print("="*60)
    
    correct = 0
    for i, ctx in enumerate(questions):
        choices = results[ctx]
        
        # Find best choice (highest logprob = least negative)
        best = max(choices.items(), key=lambda x: x[1]["logprob"])
        best_letter = best[0]
        
        print(f"\nQ{i}: Model picks {best_letter}")
        for letter, data in sorted(choices.items()):
            marker = " <--" if letter == best_letter else ""
            print(f"  {letter}: logprob={data['logprob']:.4f}{marker}")
    
    print("\n" + "="*60)
    print("To verify, compare with MMLU ground truth answers")
    print("="*60)

if __name__ == "__main__":
    main()
