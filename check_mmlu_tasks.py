
from lm_eval import tasks
from lm_eval.tasks import TaskManager

tm = TaskManager()
# Get all task names
try:
    all_tasks = tm.all_tasks
except:
    # Fallback for older API
    all_tasks = tasks.ALL_TASKS


gpqa_tasks = [t for t in all_tasks if "gpqa" in t]
print(f"Found {len(gpqa_tasks)} GPQA related tasks.")
if "gpqa" in gpqa_tasks:
    print("'gpqa' is a valid task name (likely a group).")
else:
    print("'gpqa' group not found. You may need to list them individually.")

print("All GPQA tasks:", gpqa_tasks)
