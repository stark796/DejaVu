import argparse
import json

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--output-file", type=str, default="input.jsonl")
    parser.add_argument("--task-name", type=str, default="hellaswag")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--num-data", type=int, default=None)
    args = parser.parse_args()

    seq = 1024
    total_batch = 1
    pe = "fixed"

    with open(args.output_file, "w") as f:
        pass

    class DryRunner:
        def eval(self, batch):
            with open(args.output_file, "a") as f:
                for i, text in enumerate(batch["text"]):
                    # Handle different request types
                    req_type = batch.get("request_type", ["language-model-inference"] * len(batch["text"]))[i]
                    
                    if req_type == "generate_until":
                        gen_kwargs = batch["gen_kwargs"][i]
                        max_tokens = gen_kwargs.get("max_gen_toks", gen_kwargs.get("max_new_tokens", 64))
                        stop = gen_kwargs.get("until", None)
                        
                        item = {
                            "best_of": 1,
                            "echo": False,
                            "logprobs": 1,
                            "max_tokens": max_tokens,
                            "model": "x",
                            "n": 1,
                            "prompt": text,
                            "request_type": "language-model-inference", # Can assume same type for backend or differentiate? 
                            # Usually backend just takes prompt and generation params.
                            # But wait, original code hardcoded "language-model-inference".
                            "stop": stop,
                            "temperature": gen_kwargs.get("temperature", 0),
                            "top_p": gen_kwargs.get("top_p", 1),
                        }
                    else:
                        # Loglikelihood default
                        item = {
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

                    f.write(json.dumps(item) + "\n")

            out = {
                "mask_loss": [1.0] * len(batch),
                "each_correct": [True] * len(batch),
            }
            return out

    t = DryRunner()

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    # For newer lm-eval, we need to use a TaskManager
    from lm_eval.tasks import TaskManager
    task_manager = TaskManager()
    
    task_names = args.task_name.split(",")
    task_dict = tasks.get_task_dict(
        task_names,
        task_manager=task_manager
    )

    # Set fewshot count on tasks manually if the evaluator doesn't accept it
    # Function to recursively set fewshot
    def set_fewshot(t_obj, n_shots):
        if isinstance(t_obj, dict):
            # If it looks like a task config dict (has 'task' or 'config' keys), try setting it
            if 'config' in t_obj and isinstance(t_obj['config'], dict):
                t_obj['config']['num_fewshot'] = n_shots
            elif 'num_fewshot' in t_obj:
                 t_obj['num_fewshot'] = n_shots
            
            # Recurse into values if it is a container of tasks
            for k, v in t_obj.items():
                if isinstance(v, (dict, object)) and not isinstance(v, (int, float, str, bool)):
                    set_fewshot(v, n_shots)
        else:
            # Assume it's a Task object
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
    # dumped = json.dumps(results, indent=2)
    # print(dumped)
