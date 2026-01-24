
import os
import argparse
from lm_eval import evaluator, tasks

class MockLM:
    def __init__(self):
        pass

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
    for t in task_dict.values():
         if hasattr(t, "config"):
              t.config.num_fewshot = 5
    
    print(f"Evaluating {task_name}...")
    evaluator.evaluate(lm=lm, task_dict=task_dict, limit=10)
