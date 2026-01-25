import argparse
import json
import os
import random
import numpy as np

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor

# Set fixed random seed for consistent few-shot prompt generation
# Must match the seed used in generate_task_data.py
random.seed(42)
np.random.seed(42)


def json_to_key(obj):
    return json.dumps(obj, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--task-name", type=str, default="hellaswag")
    parser.add_argument("--model-type", type=str, default="opt")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num-fewshot", type=int, default=None, help="Number of few-shot examples")
    parser.add_argument("--num-data", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Path to save the evaluation summary JSON")
    args = parser.parse_args()

    if args.model_type == "opt":
        os.environ["MODEL_NAME"] = "facebook/opt-66b"
    elif args.model_type == "bloom":
        os.environ["MODEL_NAME"] = "bigscience/bloom"
    elif args.model_type == "llama":
        os.environ["MODEL_NAME"] = "meta-llama/Llama-3.2-3B"
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    seq = 1024
    total_batch = 1
    pe = "fixed"

    class RealRunner:
        def __init__(self, args):
            self.results = {}
            self.fuzzy_results = {}  # Key by last question for fuzzy matching

            with open(args.result_file, "r") as f:
                for line in f:
                    if line.strip() == "":
                        continue

                    item = json.loads(line)
                    request = item["request"]
                    if "result" not in item:
                        continue
                    result = item["result"]

                    # Exact key match
                    self.results[json_to_key(request)] = result
                    
                    # Fuzzy key: extract last question from prompt
                    # For MCQ tasks, the last question is the actual test item
                    prompt = request.get("prompt", "")
                    # Split by "Question:" and take the last one
                    parts = prompt.split("Question:")
                    if len(parts) > 1:
                        last_question = "Question:" + parts[-1].strip()
                        self.fuzzy_results[last_question] = result

            print(f"{len(self.results)} items in the cache")
            print(f"{len(self.fuzzy_results)} fuzzy keys (by last question)")

        def _fuzzy_lookup(self, prompt):
            """Try to find result by matching last question in prompt"""
            parts = prompt.split("Question:")
            if len(parts) > 1:
                last_question = "Question:" + parts[-1].strip()
                if last_question in self.fuzzy_results:
                    return self.fuzzy_results[last_question]
            return None


        def eval(self, batch):
            from tasks.eval_harness import tokenizer

            # Detect request type
            request_type = "language-model-inference"
            if "request_type" in batch:
                if isinstance(batch["request_type"], list):
                     request_type = batch["request_type"][0]
                else:
                     request_type = batch["request_type"]

            if request_type == "generate_until":
                outputs = []
                for i, text in enumerate(batch["text"]):
                     # Reconstruct key to match generation file
                     # Note: generate_task_data uses "language-model-inference" as the stored request type even for generation
                     gen_kwargs = batch["gen_kwargs"][i]
                     max_tokens = gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", 64))
                     stop = gen_kwargs.get("until", None)
                     
                     request = {
                        "best_of": 1,
                        "echo": False,
                        "logprobs": 1,
                        "max_tokens": max_tokens,
                        "model": "x",
                        "n": 1,
                        "prompt": text,
                        "request_type": "language-model-inference", # CAUTION: Must match generate_task_data
                        "stop": stop,
                        "temperature": gen_kwargs.get("temperature", 0),
                        "top_p": gen_kwargs.get("top_p", 1),
                     }
                     key = json_to_key(request)
                     
                     if key in self.results:
                          res = self.results[key]
                          # result is [text]
                          if isinstance(res, list):
                              outputs.append(res[0])
                          else:
                              outputs.append(res) # Should be list though
                     else:
                          if args.debug:
                               print(f"MISSING KEY: {key}")
                          outputs.append("") # Dummy
                return outputs

            # Loglikelihood logic
            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch["text"]):
                request = {
                    "best_of": 1,
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 0,
                    "model": "x",
                    "n": 1,
                    "prompt": text,
                    "request_type": "language-model-inference",
                    "stop": None,
                    "temperature": 0,
                    "top_p": 1,
                }

                key = json_to_key(request)

                correct = True
                
                # Try exact match first, then fuzzy match
                result = None
                if key in self.results:
                    result = self.results[key]
                else:
                    # Try fuzzy lookup by last question
                    result = self._fuzzy_lookup(text)
                    if result and not hasattr(self, '_fuzzy_warned'):
                        print("INFO: Using fuzzy matching for prompt lookup (few-shot examples differ)")
                        self._fuzzy_warned = True

                if result is not None:

                    token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
                    tokens = result["choices"][0]["logprobs"]["tokens"]
                    top_logprobs = result["choices"][0]["logprobs"]["top_logprobs"]
                    assert token_logprobs[0] is None

                    token_ids = tokenizer.convert_tokens_to_ids(tokens)

                    obs = batch["obs"][i]
                    target = batch["target"][i]
                    eval_mask = batch["eval_mask"][i]

                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)

                    # Calculate alignment shift
                    # We assume the mismatch is due to tokenization differences (e.g. splits/merges) 
                    # but the content is the same, so we align the END of the sequences.
                    # process_request length: batch['ctx_length'][i] + 1 is WRONG (it's buffer size)
                    # We derive actual length from the mask (assuming mask covers the end).
                    
                    try:
                        # indices where mask is True
                        true_indices = [idx for idx, m in enumerate(eval_mask) if m]
                        if true_indices:
                            last_mask_idx = true_indices[-1]
                            # last_mask_idx corresponds to all_tokens[last_mask_idx + 1]
                            # so len(all_tokens) = last_mask_idx + 1 + 1 = last_mask_idx + 2
                            proc_len = last_mask_idx + 2
                        else:
                            # Fallback if mask is empty (should not happen for valid tasks)
                            proc_len = batch['ctx_length'][i] 
                    except:
                        proc_len = len(tokens) # Default to no shift if calculation fails

                    inf_len = len(tokens)
                    shift = inf_len - proc_len
                    
                    if args.debug and i < 4:
                         print(f"DEBUG: derived proc_len from mask: {proc_len}, inf_len: {inf_len}, shift: {shift}")
                    
                    for token_idx, mask_val in enumerate(eval_mask):
                        if not mask_val:
                            continue

                        # Map index: we want the token at (token_idx + 1) in the process_request frame
                        # So we map it to inference frame.
                        mapped_idx = token_idx + 1 + shift
                        
                        if mapped_idx < 0 or mapped_idx >= len(tokens):
                             # This might happen if truncation differs significantly
                             continue

                        sum_lobprob += token_logprobs[mapped_idx]
                        n_positive += 1
                        
                        # VERIFY: Print what we are scoring
                        # Only for first few requests to avoid spam
                        if i < 4: 
                            print(f"DEBUG: Req {i}")
                            print(f"  proc_len: {proc_len}, inf_len: {inf_len}, shift: {shift}")
                            print(f"  token_idx: {token_idx} -> mapped_idx: {mapped_idx}")
                            print(f"  Scoring Token: '{tokens[mapped_idx]}'")
                            print(f"  Logprob: {token_logprobs[mapped_idx]}")
                            try:
                                print(f"  Tokens[-5:]: {tokens[-5:]}")
                            except:
                                pass

                        # Only check correctness if we have logprobs
                        try:
                             if top_logprobs and mapped_idx < len(top_logprobs):
                                correct = correct and (
                                    tokens[mapped_idx]
                                    == next(iter(top_logprobs[mapped_idx].keys()))
                                )
                        except Exception:
                             pass
                    
                    mask_loss.append(-sum_lobprob)

                    each_correct.append(correct)

                else:
                    if args.debug:
                        print(f"FATAL: Missing key for request inside RealRunner.eval")
                        print(f"Constructed Request: {json.dumps(request, indent=2)}")
                        print(f"Constructed Key: {key}")
                        print(f"Total keys in results: {len(self.results)}")
                        print("First 3 keys in results:")
                        for k in list(self.results.keys())[:3]:
                            print(k)
                    # Skip missing keys (e.g., prompts that were skipped during inference due to OOM)
                    if not hasattr(self, '_missing_warned'):
                        print(f"WARNING: Skipping {1} prompt(s) with missing results (possibly OOM during inference)")
                        self._missing_warned = True
                    self._missing_count = getattr(self, '_missing_count', 0) + 1
                    mask_loss.append(float('inf'))  # High loss for missing
                    each_correct.append(False)  # Mark as incorrect

            out = {
                "mask_loss": mask_loss,
                "each_correct": each_correct,
            }

            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    # For newer lm-eval, we need to use a TaskManager
    from lm_eval.tasks import TaskManager
    task_manager = TaskManager()

    # Support comma-separated task names
    task_names = args.task_name.split(",")
    task_dict = tasks.get_task_dict(
        task_names,
        task_manager=task_manager
    )

    # Function to recursively set fewshot (copied from generate_task_data.py)
    def set_fewshot(t_obj, n_shots):
        if isinstance(t_obj, dict):
            if 'config' in t_obj and isinstance(t_obj['config'], dict):
                t_obj['config']['num_fewshot'] = n_shots
            elif 'num_fewshot' in t_obj:
                 t_obj['num_fewshot'] = n_shots
            for k, v in t_obj.items():
                if isinstance(v, (dict, object)) and not isinstance(v, (int, float, str, bool)):
                    set_fewshot(v, n_shots)
        else:
            if hasattr(t_obj, 'config') and hasattr(t_obj.config, 'num_fewshot'):
                t_obj.config.num_fewshot = n_shots
            try:
                t_obj.num_fewshot = n_shots
            except AttributeError:
                pass
            if hasattr(t_obj, 'set_num_fewshot'):
                t_obj.set_num_fewshot(n_shots)

    if args.num_fewshot is not None:
        for task_name, task_obj in task_dict.items():
            set_fewshot(task_obj, args.num_fewshot)

    results = evaluator.evaluate(
        lm=adaptor,
        task_dict=task_dict,
        limit=args.num_data,
    )

    dumped = json.dumps(results, indent=2)
    print(json.dumps(results, indent=2))
    
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    if "results" in results:
        print(json.dumps(results["results"], indent=2))
    else:
        print("No 'results' key found in output.")
    print("="*40)

    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(dumped)
        print(f"Results saved to {args.output_json}")
