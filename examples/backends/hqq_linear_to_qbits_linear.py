# Support List:
# - [x] `gropu_size` is -1
# - [ ] `group_size` is not -1
# - [x] bits: 4
# - [ ] other bits

import hqq
import hqq.backends
import hqq.backends.qbits
import hqq.core.quantize as hqq_quant
import torch
import copy

N = 1
bs = 32
in_features = 512 * N
out_features = 1024 * N
device = "cpu"
HAS_BIAS = False
float_lin = torch.nn.Linear(in_features, out_features, bias=HAS_BIAS)
compute_dtype = torch.float32
group_size = 64
AXIS = 1

# Quantization settings
quant_config = hqq_quant.BaseQuantizeConfig(nbits=4, group_size=group_size, quant_zero=False, quant_scale=False, axis=AXIS)

#Set default backends, to compare with int4mm
if(quant_config['weight_quant_params']['axis']==0):
    hqq_quant.HQQLinear.set_backend(hqq_quant.HQQBackend.ATEN)
else:
    hqq_quant.HQQLinear.set_backend(hqq_quant.HQQBackend.PYTORCH_FORWARD)

print(f"hqq_quant.HQQLinear.backend: {hqq_quant.HQQLinear.backend}")

# Replace your linear layer
hqq_layer = hqq_quant.HQQLinear(
    float_lin,  # torch.nn.Linear or None
    quant_config=quant_config,  # quantization configuration
    compute_dtype=compute_dtype,  # compute dtype
    device=device,  # cuda device
    initialize=True,  # Use False to quantize later
    del_orig=True,  # if True, delete the original layer
)

sample_input = torch.randn(bs, in_features)
print(hqq_layer)
print(hqq_layer.W_q.shape)


def compare_two_tensors(a, b, atol=1e-5):
    if a.shape != b.shape:
        print(f"Shapes don't match: {a.shape} != {b.shape}")
        return False
    diff = torch.abs(a - b)
    max_diff = diff.max()
    if max_diff > atol:
        print(f"Max diff: {max_diff}")
        return False
    print("Tensors are equal")
    return True

@torch.no_grad()
def check_correctness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    import numpy as np
    float_lin = torch.nn.Linear(in_features, out_features, bias=HAS_BIAS)
    # print(f"weight shape: {float_lin.weight}")
    compute_dtype = torch.float32
    # Quantization settings
    quant_config = hqq_quant.BaseQuantizeConfig(nbits=4, group_size=group_size, quant_zero=False, quant_scale=False, axis=AXIS)

    # Replace your linear layer
    hqq_layer = hqq_quant.HQQLinear(
        float_lin,  # torch.nn.Linear or None
        quant_config=quant_config,  # quantization configuration
        compute_dtype=compute_dtype,  # compute dtype
        device=device,  # cuda device
        initialize=True,  # Use False to quantize later
        del_orig=True,  # if True, delete the original layer
    )
    hqq_out = hqq_layer(copy.deepcopy(sample_input))
    qbit_linear1 = hqq.backends.qbits.patch_hqq_to_qbits(copy.deepcopy(hqq_layer))
    qbit_linear2 = hqq.backends.qbits.patch_hqq_to_qbits(copy.deepcopy(hqq_layer))
    print(qbit_linear2)
    qbit_out1 = qbit_linear1(copy.deepcopy(sample_input))
    qbit_out2 = qbit_linear2(copy.deepcopy(sample_input))
    if not compare_two_tensors(qbit_out2, qbit_out1):
        print(f"!!!!!!! Seed: {seed} failed")
    if not compare_two_tensors(hqq_out, qbit_out1):
        print(f"!!!!!!! Seed: {seed} failed")
        import pdb; pdb.set_trace()
    else:
        print(f"Seed: {seed} passed")

# check_correctness(800)
check_correctness(301)
check_correctness(301)
check_correctness(301)
check_correctness(301)
for seed in torch.range(0, 1000, 100):
    check_correctness(seed)

# hqq_out = hqq_layer(sample_input)
# import copy

# qbit_linear = hqq.backends.qbits.patch_hqq_to_qbits(copy.deepcopy(hqq_layer))

# print(qbit_linear)

# qbit_out = qbit_linear(sample_input)
# diff = torch.abs(hqq_out - qbit_out).max()
# compare_two_tensors(hqq_out, qbit_out)


def bench_fn(func, times=100):
    import time

    start = time.time()
    for i in range(times):
        func()
    end = time.time()
    avg = (end - start) / times
    print(avg)
    return avg


# Max diff: 6.243084813196034e+32
# 0.007208244800567627
# 0.05162894010543823
# 0.0016379785537719727

# with torch.no_grad():
#     def bench():
#         bench_fn(lambda: float_lin(sample_input))
#         bench_fn(lambda : hqq_layer(sample_input))
#         bench_fn(lambda : qbit_linear(sample_input))

#     bench()
