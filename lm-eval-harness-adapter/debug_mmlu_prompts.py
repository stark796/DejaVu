
import os
import argparse
try:
    from lm_eval.api.model import LM  # lm-eval >= 0.4.0
except ImportError:
    from lm_eval.base import LM  # lm-eval < 0.4.0
from lm_eval import evaluator, tasks

class MockLM(LM):
    def __init__(self):
        super().__init__()

    def loglikelihood(self, requests):
        print("\n--- DEBUGGING REQUESTS ---")
        for i, req in enumerate(requests[:5]):  # Print first 5
            if hasattr(req, "args"):
                ctx, cont = req.args
            else:
                ctx, cont = req
            
            print(f"Request {i}:")
            print(f"Context (last 100 chars): ...{ctx[-100:]}")
            print(f"Continuation: '{cont}'")
            print("-" * 20)
        
        # Return dummy results
        return [(-1.0, True)] * len(requests)

    def loglikelihood_rolling(self, requests):
         raise NotImplementedError

    def generate_until(self, requests):
         raise NotImplementedError

if __name__ == "__main__":
    lm = MockLM()
    
    # Simple task manager setup for newer lm-eval
    from lm_eval.tasks import TaskManager
    task_manager = TaskManager()
    
    task_name = "mmlu_business_ethics"
    task_dict = tasks.get_task_dict([task_name], task_manager=task_manager)
    
    # Set 5-shot
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

    for task_obj in task_dict.values():
         set_fewshot(task_obj, 5)
    
    print(f"Evaluating {task_name}...")
    evaluator.evaluate(lm=lm, task_dict=task_dict, limit=10)
