# DeepSeek-V4.1-Flash-1017 NVFP4 on 4× DGX Spark

Runtime overlay for [`nvidia/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4)
served by stock `vllm/vllm-openai:v0.32.0` on four DGX Spark (GB10) nodes, TP=4 over the
ConnectX-7 fabric.

The recipe in [lmdeck](https://github.com/javierbits/lmdeck) mounts
`patches/engram_mxfp4.py` over the installed vLLM file. There is no image to build or push.

## What the overlay is

NVIDIA's checkpoint stores the two Engram tables as MXFP4 (`engram_embed_quant: "mxfp4"`).
Stock vLLM 0.32.0 reads Engram tables as FP8 and fails at load. The fix is
[vllm-project/vllm#61530](https://github.com/vllm-project/vllm/pull/61530) (open at the time
of writing): a Triton-kernel nibble decode plus a packed row layout, so a TP=4 rank stores
12.16 GiB per table instead of 23.60 GiB.

`patches/engram_mxfp4.py` is vLLM **v0.32.0**'s
`vllm/models/deepseek_v41/common/engram.py` with that PR applied (the PR's base commit,
`87e7dce3a79e3a6d0944f890a61f2970f8be0788`, has the same file content as the v0.32.0 tag).
`patches/0001-mxfp4-engram.patch` is the exact diff for provenance. The file is mounted
read-only at:

```
/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/engram.py
```

The rest of the stack is the stock vLLM image, so the FlashInfer sparse-MLA kernels,
tokenizer, parser and scheduler are the vendor's.

## Version notes

- NVIDIA's model card reports `vllm/vllm-openai:v0.31.0` with "that PR's Engram changes
  applied". The PR was written against v0.32.0's Engram file; v0.31.0's file is a different,
  smaller implementation and the diff does not apply. v0.32.0 is the first release whose
  file matches the PR base, so that is the image pinned here.
- The PR is not merged upstream. A newer stock image will not load this checkpoint until it
  is.
- SGLang is not validated for this checkpoint.

## Memory on a GB10

The checkpoint is 399.55 GiB (429 GB) in 48 safetensors shards:

| part | total | per rank at TP=4 |
|---|---:|---:|
| non-Engram weights | 302.23 GiB | 75.56 GiB |
| Engram tables, MXFP4 | 97.28 GiB | 24.32 GiB |
| **resident total** | **399.55 GiB** | **99.88 GiB** |

A GB10 exposes ~121.7 GiB to CUDA. Host-pinned `cpu_offload` is not a second tier: CPU and
GPU share the same physical pool. 99.88 GiB leaves ~21.8 GiB for KV, activations and
loader peaks. That is why the profile starts at 8K context, one sequence, eager execution,
and `--gpu-memory-utilization 0.75`; raise one knob at a time and watch `MemAvailable` on
every node. Long context is a capacity question on this board, not a flag.

## Reproduce without lmdeck

```bash
docker pull vllm/vllm-openai:v0.32.0
git clone https://github.com/javierbits/DeepSeek-V4.1-Flash-1017-NVFP4-DGX-Spark-recipe
hf download nvidia/DeepSeek-V4.1-Flash-1017-NVFP4 \
  --revision 480550d349ede47a048eb65a276c5c01a06256e2
```

Then run `vllm serve` on each node with `--nnodes 4 --node-rank 0..3`, `--headless` on
ranks 1-3, `--tensor-parallel-size 4`, `--master-addr <head CX7 IP>`, and the Engram,
attention, KV and tokenizer flags in the lmdeck recipe. The overlay file above must be
mounted into every container.

## License

The overlay is vLLM source, Apache-2.0. The recipe text and scripts here are MIT.
DeepSeek-V4.1-Flash-1017 and NVIDIA's NVFP4 conversion are MIT.
