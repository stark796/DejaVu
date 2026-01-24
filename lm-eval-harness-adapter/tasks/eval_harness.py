from functools import partial

import os
import transformers
try:
    from lm_eval.api.model import LM  # lm-eval >= 0.4.0
except ImportError:
    from lm_eval.base import LM  # lm-eval < 0.4.0
from tqdm import tqdm
import numpy as np

from tasks.util import sample_batch, shrink_seq
import multiprocessing
import ftfy

tokenizer = None


def process_init():
    global tokenizer
    model_name = os.environ.get("MODEL_NAME", "facebook/opt-1.3b")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = True


#     tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
#     tokenizer.model_max_length = int(1e30)
#     tokenizer.pad_token = "<|endoftext|>"

#     assert tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373]


def process_request(x, seq):
    global tokenizer

    # Handle Instance objects from newer lm-eval
    if hasattr(x, "args"):
        ctx, cont = x.args
    else:
        ctx, cont = x

    #     ctx_tokens = tokenizer.encode("<|endoftext|>" + ftfy.fix_text(ctx, normalization="NFKC"))
    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    cont_text = ftfy.fix_text(cont, normalization="NFKC")
    all_text = ctx_text + cont_text

    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]
    cont_tokens = tokenizer(cont_text, add_special_tokens=False)["input_ids"]

    all_tokens = ctx_tokens + cont_tokens
    all_tokens = np.array(all_tokens)[-seq:]  # truncate sequence at seq length

    provided_ctx = len(all_tokens) - 1
    pad_amount = seq - provided_ctx

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)

    return {
        "obs": np.pad(
            all_tokens[:-1], ((0, pad_amount),), constant_values=pad_id
        ),
        "target": np.pad(
            all_tokens[1:], ((0, pad_amount),), constant_values=pad_id
        ),
        "ctx_length": seq,
        "eval_mask": np.logical_and(
            np.arange(0, seq) > len(all_tokens) - len(cont_tokens) - 2,
            np.arange(0, seq) < len(all_tokens) - 1,
        ),
        "prompt": ctx_text,
        "target": cont_text,
        "text": all_text,
    }



def process_request_gen(x, seq):
    global tokenizer
    
    # Handle Instance objects from newer lm-eval
    if hasattr(x, "args"):
        ctx, gen_kwargs = x.args
    else:
        ctx, gen_kwargs = x

    ctx_text = ftfy.fix_text(ctx, normalization="NFKC")
    ctx_tokens = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]
    
    # For generation, we don't know the full length yet, but we truncate context if needed
    # Usually we leave some room for generation.
    # The benchmark_inference script handles generation, so here we just prepare the prompt.
    
    return {
        "prompt": ctx_text,
        "gen_kwargs": gen_kwargs,
        "text": ctx_text, # For consistency
        "request_type": "generate_until"
    }


class EvalHarnessAdaptor(LM):
    def greedy_until(self, requests):
        raise Exception("unimplemented")

    def generate_until(self, requests):
        """Supported method for newer lm-eval versions (GPQA etc)"""
        # Convert requests
        # We cannot easily parallelize with map/imap if using different seq? 
        # Actually generate_until requests are simpler.
        
        # Note: requests is a list of Instances or tuples (ctx, gen_kwargs)
        res = []
        for req in requests:
           res.append(process_request_gen(req, self.seq))
           
        # We pass the list to DryRunner's eval (or generate)
        # Assuming tpu.eval can handle it or we update it.
        # DryRunner expects a "batch", so we wrap it.
        
        # Just process them individually or in batch?
        # modify sample_batch to handle this?
        
        # To match existing flow, we pass to self.tpu.eval
        # But DryRunner eval writes to file.
        
        # Let's chunk it by self.batch
        outputs = []
        for i in range(0, len(res), self.batch):
            batch = res[i : i + self.batch]
            # Convert list of dicts to dict of lists
            batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
            
            # Call DryRunner
            batch_outputs = self.tpu.eval(batch_dict)
            
            # If tpu.eval returns a list (RealRunner for generation), extend outputs
            # If tpu.eval returns dict (DryRunner), we might need to handle it?
            # DryRunner for generation needs to return something? 
            # In generate_task_data, DryRunner.eval returns validation dict.
            # But for generation, we probably want it to return dummy strings if we are generating data?
            
            if isinstance(batch_outputs, list):
                outputs.extend(batch_outputs)
            elif isinstance(batch_outputs, dict) and "generated_text" in batch_outputs:
                 outputs.extend(batch_outputs["generated_text"])
            else:
                 # Fallback for DryRunner during data generation
                 for _ in batch:
                    outputs.append("dummy response")
                
        return outputs

    def loglikelihood_rolling(self, requests):
        raise Exception("unimplemented")

    def __init__(self, tpu_cluster, seq, batch, shrink, min_seq=None):
        super().__init__()
        self.tpu = tpu_cluster
        self.seq = seq
        self.batch = batch
        self.shrink = shrink
        self.min_seq = min_seq

        self.pool = multiprocessing.Pool(initializer=process_init)
        process_init()

    def convert_requests(self, requests):
        return self.pool.imap(partial(process_request, seq=self.seq), requests)

    def loglikelihood(self, requests):
        output = []

        r = self.convert_requests(requests)
        zero_example = process_request(requests[0], self.seq)

        for b in tqdm(
            sample_batch(r, self.batch, zero_example),
            desc="LM eval harness",
            total=len(requests) // self.batch,
        ):
            if self.shrink:
                b = shrink_seq(b, min_seq=self.min_seq)

            out = self.tpu.eval(b)

            for loss, correct in zip(out["mask_loss"], out["each_correct"]):
                output.append((float(-loss), bool(correct)))

        return output

