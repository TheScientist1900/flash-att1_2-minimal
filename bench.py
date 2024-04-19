import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.

shapes = [
    [16, 12, 64, 64],
    [16, 12, 256, 64],
    [16, 12, 2048, 64],
]

for shape in shapes:
    batch_size, n_head, seq_len, head_embd = shape

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    # Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
    def manual_attn(q, k, v):
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return y
    
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        manual_result = manual_attn(q, k, v)
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    print('=== profiling manual attention ===')
    print(shape)
    print(prof.total_average().self_cpu_time_total_str)
    print(prof.total_average().self_cuda_time_total_str)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        my_result = minimal_attn.myForward(q, k, v)
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    print('=== profiling flash attention 1===')
    print(shape)
    print(prof.total_average().self_cpu_time_total_str)
    print(prof.total_average().self_cuda_time_total_str)
    print('flash attention 1 values sanity check:', torch.allclose(my_result, manual_result, rtol=0, atol=1e-02))
    continue
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        my_result = minimal_attn.myForward2(q, k, v)
    # print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    
    print('=== profiling flash attention 2===')
    print(shape)
    print(prof.total_average().self_cpu_time_total_str)
    print(prof.total_average().self_cuda_time_total_str)
    print('flash attention 2 values sanity check:', torch.allclose(my_result, manual_result, rtol=0, atol=1e-02))
