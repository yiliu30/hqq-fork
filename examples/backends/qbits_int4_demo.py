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
BACKEND = "qbits"

os.environ["TOKENIZERS_PARALLELISM"]  = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

cache_path     = '.'
# model_id       = "/models/Llama-2-7b-chat-hf/"
# model_id = "/models/TinyLlama-1.1B-Chat-v1.0"
model_id = "/mnt/disk4/modelHub/Llama-2-7b-chat-hf/"
model_id = "/mnt/disk4/modelHub/TinyLlama-1.1B-Chat-v1.0/"
# model_id = "/models/opt-125m"
compute_dtype  = torch.float32 #int4 kernel only works with float16
# device         = 'cuda:0'
device = "cpu"
from tiny_benchmark import benchmodel
group_size = 64

##########################################################################################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import *

tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
model        = HQQModelForCausalLM.from_pretrained(model_id, cache_dir=cache_path, torch_dtype=compute_dtype, )#attn_implementation="sdpa")
quant_config = BaseQuantizeConfig(nbits=4, group_size=group_size, quant_scale=False, quant_zero=False, axis=1)
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
prepare_for_inference(model, backend=BACKEND, verbose=True)

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

# with torch.no_grad():
#     # @dump_func_time
#     # def test_gen():
#     #     #Generate
#     #     gen = HFGenerator(model, tokenizer, max_new_tokens=20, do_sample=True, compile=None) 

#     #     out = gen.generate("Write an essay about large language models.", print_tokens=True)
#     #     out = gen.generate("Tell me a funny joke!", print_tokens=True)
#     #     out = gen.generate("How to make a yummy chocolate cake?", print_tokens=True)
        
#     # test_gen()
#     @dump_func_time
#     def gen():
#         prompt = "Today I believe we can finally"
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#         # generate up to 30 tokens
#         outputs = model.generate(input_ids.to(model.device), do_sample=False, max_length=30)
#         kk = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         print(f"gen {kk}")
#     gen()

# ==-------------------------------------------------------------------------==
# Benchmark hqq + qbits
# ==-------------------------------------------------------------------------==
import sys
sys.path.append("./")
# from .tiny_benchmark import benchmodel

benchmodel(model_and_tokenizer=(model, tokenizer), pretrained=True, batch_size=4)


# ==-------------------------------------------------------------------------==
# Benchmark hqq
# ==-------------------------------------------------------------------------==
"""
- Default
|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)   |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:----------------|
|            1 |               32 |              32 |              54.68 |              2.18 | 1.87 GB (0.00%) |
|            1 |               64 |              64 |             104.67 |              2.18 | 1.91 GB (0.00%) |
|            1 |              128 |             128 |             192.44 |              2.04 | 2.13 GB (0.00%) |


Version: Qbits
|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)   |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:----------------|
|            1 |               32 |              32 |             241.11 |              1.98 | 2.72 GB (0.00%) |
|            1 |               64 |              64 |             395.94 |              1.93 | 2.76 GB (0.00%) |
|            1 |              128 |             128 |             570.89 |              1.79 | 2.89 GB (0.00%) |

Version: Qbits
|   Batch Size |   Prefill Length |   Decode Length |   Prefill tokens/s |   Decode tokens/s | Memory (VRAM)   |
|-------------:|-----------------:|----------------:|-------------------:|------------------:|:----------------|
|            4 |               32 |              32 |             554.12 |             75.58 | 2.87 GB (0.00%) |
|            4 |               64 |              64 |             718.42 |             75.33 | 2.93 GB (0.00%) |
|            4 |              128 |             128 |             827.39 |             73.95 | 3.00 GB (0.00%) |
|            4 |              256 |             256 |             786.73 |             72.56 | 3.06 GB (0.00%) |
|            4 |              512 |             512 |             780.28 |             72.82 | 3.18 GB (0.00%) |
|            4 |             1024 |            1024 |             724.8  |             74.46 | 3.16 GB (0.00%) |
|            4 |             2048 |            2048 |             697.49 |             74.26 | 3.18 GB (0.00%) |
# ==-------------------------------------------------------------------------==
#  The output
# ==-------------------------------------------------------------------------==
"""
"""

QBitsLinear(in_features=5632, out_features=2048, bias=False, group_size=64)
gen ["Today I believe we can finally see the light at the end of the tunnel.\nI'm not sure if I'm ready to accept"]
gen took 1.585073471069336 seconds
gen ["Today I believe we can finally see the light at the end of the tunnel.\nI'm not sure if I'm ready to accept"]
gen took 29.611412048339844 seconds

The qbits kernel
/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
<URI leng leng lengramramni lengwaldram lengramni vor vorram vor vor vor< leng lengni nabram program lengram vorramram program lengwald program leng leng leng leng<URIURI vorni nab lengram programram vor leng leng lengramni nabram lengramtest_gen took 129.47010707855225 seconds
/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
 obviously hopefully everybody obviously obviously obviously hopefully nobody hopefully obviously nobody hopefully hopefully hopefully hopefully hopefully everybody nobody nobody obviously nobody hopefully obviously hopefully obviously nobody everybody surely everybody obviously everybody hopefully hopefully everybody hopefully hopefully obviously nobody hopefully obviously obviously nobody hopefully obviously hopefully nobody obviously hopefully nobody hopefully hopefully hopefully hopefully obviously obviously hopefully hopefullytest_gen took 6.7808942794799805 seconds

# The default kernel
 obviously hopefully everybody obviously obviously obviously hopefully nobody hopefully obviously nobody hopefully hopefully hopefully hopefully hopefully everybody nobody nobody obviously nobody hopefully obviously hopefully obviously nobody everybody surely everybody obviously everybody hopefully hopefully everybody hopefully hopefully obviously nobody hopefully obviously obviously nobody hopefully obviously hopefully nobody obviously hopefully nobody hopefully hopefully hopefully hopefully obviously obviously hopefully hopefullytest_gen took 1016.4369328022003 seconds


=== v1.0
qbits,
/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
< „  née „ heten lengheten „value „ „ „ „ leng lengheten vor<issue lengéo nabwerke „ „née nabalogalogéoéo nab nabheten vorheten vor< „néeheten vor voréo nabalogéo „ „    néeéoheten vortest_gen took 4.646540880203247 seconds


default
< „  née „ heten lengheten „value „ „ „ „ leng lengheten vor<issue lengéo nabwerke „ „née nabalogalogéoéo nab nabheten vorheten vor< „néeheten vor voréo nabalogéo „ „    néeéoheten vortest_gen took 71.04186415672302 seconds


/home/yliu7/miniconda3/envs/cu121/lib/python3.9/site-packages/torch/backends/cuda/__init__.py:342: FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed. Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
  warnings.warn(
 Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Einzeln Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung sierp Begriffe Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung sierp Unterscheidung Hinweis Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidungtest_gen took 10.458760976791382 seconds


 Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Einzeln Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung sierp Begriffe Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung sierp Unterscheidung Hinweis Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidung Unterscheidungtest_gen took 625.4215815067291 seconds

"""