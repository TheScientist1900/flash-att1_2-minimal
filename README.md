# flash-attention-minimal
- Modify from [flash-attention-minimal
](https://github.com/tspeterkim/flash-attention-minimal.git)
- Add flash attention 2
## Usage
### Prerequisite
* PyTorch (with CUDA)
* `Ninja` for loading in C++

### Benchmark
Compare the wall-clock time between manual attention and minimal flash attention:
```
python bench.py
```

### Analysis
- Flops/Extra memory/IO complexity in `analyze.md`
- The formula used in flash attention in `flashattn-note.pdf`
