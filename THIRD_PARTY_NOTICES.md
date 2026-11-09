# Third-party notices

Everything not listed below is MIT ([LICENSE](LICENSE)) and was written for these builds.

| path | license | origin |
|---|---|---|
| `overlays/vllm-tp6/*`, `overlays/vllm-tp4/*`, `overlays/vllm-common/*`, `image/r17-quantization/*` | Apache-2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)) | vLLM source files (vLLM project contributors) as shipped in the base image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907` and in the r17 release of [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) (pins in `image/pins/vllm.lock.json`), modified for these builds: EXL3 quantization hooks, MTP `eh_proj` fix, DCP2 sparse-MLA, DCP-aware MTP draft slots, FULL-graph warmup |
| `overlays/d3-roce/vllm/*` | Apache-2.0 | vLLM files from the same base image plus the RoCE all-reduce adapter ported from [local-inference-lab/vllm#597](https://github.com/local-inference-lab/vllm/pull/597) (author @original-el8); change notes are in the header of `b12x_roce_all_reduce.py` |
| `overlays/d3-roce/b12x_comm_roce/*` | Apache-2.0 | `b12x.comm.roce` ("RoCEnante") from [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (@original-el8, Luke Alonso @lukealonso), API version 1, unmodified |
| `overlays/tp6-fastload/weight_utils_filter.py`, `overlays/tp4-load-accel/ep_weight_filter.py`, `overlays/tp4-load-accel/exl3_cached.py`, `gptq/tp6/mxfp8_utils.py` | Apache-2.0 | vLLM files, modified (load filters, prepared-weight cache) or unmodified (`mxfp8_utils.py`, used by the solver's weight-error check) |
| `experiments/d13-adaptive-k/overlay/*` | Apache-2.0 | vLLM files, modified (env-gated extra FULL decode graphs) |
| `experiments/d13-adaptive-k/adaptive_k_phase.py` | Apache-2.0 | port of Matt Mastracci's adaptive-k scheduler from [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) (`experimental/adaptive-k/adaptive_k.py`) to the MTP drafter, plus per-phase state |
| `third_party/e3/*` (except `vendor/`) | **AGPL-3.0** ([third_party/e3/LICENSE.mia](third_party/e3/LICENSE.mia), also [LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt)) | derived from MiaAI-Lab's E3 kernel, [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks @ 9348755](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/9348755653f6f8cda5d56562c05462724c40fcbd/overlay) (`exl3_fat_moe.cu`), extended for K3/K4 per-expert dispatch and the TP4/TP6 fragment layouts. Kept under its own license; it is loaded as a separate runtime package. `grouped_fragments.cubin` is built from `grouped_fragments.cu` (CUDA 13.0.88, sm_121a): `nvcc -O3 -lineinfo --cubin -arch=sm_121a -I $CUDA/include e3/grouped_fragments.cu -o e3/grouped_fragments.cubin`. The shipped cubin was rebuilt from a neutral path so no build paths are embedded; its code, constant and info sections are byte-identical to the served one |
| `third_party/e3/vendor/*` | MIT ([third_party/e3/vendor/LICENSE.exllamav3](third_party/e3/vendor/LICENSE.exllamav3)) | headers from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) 0.0.43; `util.cuh` drops unused host cuBLAS helpers |

## Referenced, not included

- **b12x** (Apache-2.0): the images vendor the r17 b12x package, built from
  [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) base `c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5`
  plus the r17 release integration patch (SHA256 `9dde29254d7db866f189eea416c1c2de939d7e6e804d4d8f594368a701b12cf2`) from
  [local-inference-lab/blackwell-llm-docker @ 64f5c35](https://github.com/local-inference-lab/blackwell-llm-docker/tree/64f5c3503ecb8d3d315340c8699a18266af9586c/patches/releases/infernal-invocation-r17).
  Full lock: `image/pins/b12x.lock.json`.
- **Base image** `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907` (MiaAI-Lab): vLLM `0.1.dev20051+g487ecf187`,
  torch `2.13.0+cu130`, exllamav3 `0.0.43`, CUTLASS DSL `4.6.2`.
- **Weights**: [davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) (a native TP4 export of GLM 5.3).
- **GB10 prefill fix** in `image/patch_runtime.py` (persistent top-k off): from
  [drowzeys/keys-GLM-5.3-EXL3](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3/blob/main/serving/Dockerfile.keys).
- **Decode prompt harness** used by `eval/accept.py`: adapted from Matt Mastracci's
  [glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10) `dev/repro/decode.py`; not included.

## Credits

- Matt Mastracci ([@mmastrac](https://github.com/mmastrac)): the adaptive-k scheduler we ported (D13), the decode
  benchmark harness our acceptance numbers use, and the megamoe prefill kernel whose structure the
  `experiments/megamoe` prototype follows (no code copied).
- local-inference-lab: b12x, the RoCE one-shot collectives and the r17 EXL3 loader.
- MiaAI-Lab: the ARM base image and the E3 kernel.
- davidsyoung: the EXL3 TR3 3.25 bpw quantization.
