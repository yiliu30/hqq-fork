##########################################################################################################################################################
# Fixed random seed for reproducibility
##########################################################################################################################################################
seed = 0
import random
random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)

# pip install git+https://github.com/mobiusml/hqq.git;
# pip install git+https://github.com/IST-DASLab/marlin.git;
# num_threads=12; OMP_NUM_THREADS=$num_threads CUDA_VISIBLE_DEVICES=0 ipython3 
##########################################################################################################################################################
import torch, os

BACKEND = "default" 

os.environ["TOKENIZERS_PARALLELISM"]  = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

cache_path     = '.'
model_id       = "/models/Llama-2-7b-chat-hf/"
# model_id = "/models/TinyLlama-1.1B-Chat-v1.0"
# model_id = "/models/opt-125m"
compute_dtype  = torch.float32 #int4 kernel only works with float16
# device         = 'cuda:0'
device = "cpu"

##########################################################################################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import *

tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
model        = HQQModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, attn_implementation="sdpa")
quant_config = BaseQuantizeConfig(nbits=4, group_size=None, quant_scale=False, quant_zero=False, axis=1)
print(model)
model.quantize_model(quant_config=quant_config, compute_dtype=compute_dtype, device=device)

#Set default backends, to compare with int4mm
if(quant_config['weight_quant_params']['axis']==0):
    HQQLinear.set_backend(HQQBackend.ATEN)
else:
    HQQLinear.set_backend(HQQBackend.PYTORCH)

##########################################################################################################################################################

#Replace HQQLinear layers matmuls to support int4 mm
from hqq.utils.patching import prepare_for_inference
prepare_for_inference(model, backend=BACKEND)

#Import custom HF generator
from hqq.utils.generation_hf import HFGenerator


def dump_func_time(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start} seconds")
        return result
    return wrapper

with torch.no_grad():
    @dump_func_time
    def test_gen():
        #Generate
        gen = HFGenerator(model, tokenizer, max_new_tokens=20, do_sample=True, compile=None) 

        out = gen.generate("Write an essay about large language models.", print_tokens=True)
        out = gen.generate("Tell me a funny joke!", print_tokens=True)
        out = gen.generate("How to make a yummy chocolate cake?", print_tokens=True)
        
    test_gen()
    


# ==-------------------------------------------------------------------------==
#  The output
# ==-------------------------------------------------------------------------==
"""

The qbits kernel
/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
<URI leng leng lengramramni lengwaldram lengramni vor vorram vor vor vor< leng lengni nabram program lengram vorramram program lengwald program leng leng leng leng<URIURI vorni nab lengram programram vor leng leng lengramni nabram lengramtest_gen took 129.47010707855225 seconds
/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
 obviously hopefully everybody obviously obviously obviously hopefully nobody hopefully obviously nobody hopefully hopefully hopefully hopefully hopefully everybody nobody nobody obviously nobody hopefully obviously hopefully obviously nobody everybody surely everybody obviously everybody hopefully hopefully everybody hopefully hopefully obviously nobody hopefully obviously obviously nobody hopefully obviously hopefully nobody obviously hopefully nobody hopefully hopefully hopefully hopefully obviously obviously hopefully hopefullytest_gen took 6.7808942794799805 seconds

# The default kernel
 obviously hopefully everybody obviously obviously obviously hopefully nobody hopefully obviously nobody hopefully hopefully hopefully hopefully hopefully everybody nobody nobody obviously nobody hopefully obviously hopefully obviously nobody everybody surely everybody obviously everybody hopefully hopefully everybody hopefully hopefully obviously nobody hopefully obviously obviously nobody hopefully obviously hopefully nobody obviously hopefully nobody hopefully hopefully hopefully hopefully obviously obviously hopefully hopefullytest_gen took 1016.4369328022003 seconds

"""