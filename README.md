# GLM 5.3 (full) EXL3 on DGX Spark: TP6 and TP4 builds

> **Newer: [gr8](https://github.com/kindlingai/gr8)** serves the full GLM-5.3 on three to six DGX Sparks with its
> own CUDA engine, without vLLM, and has the current numbers. This repo is the earlier vLLM build, kept for
> reference.

Two ways we serve the full GLM 5.3 model (MoE, 256 experts, MLA + sparse attention, one MTP layer) from the
[davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw) EXL3
quantization with vLLM on NVIDIA DGX Spark (GB10) nodes:

| | TP6 | TP4 |
|---|---|---|
| nodes | 6 × DGX Spark | 4 × DGX Spark |
| weights per node (planning estimate, not re-measured) | ~83 GB (experts re-fragmented six ways) | ~96 GB (native TP4 export) |
| prefill 8K / 32K (tok/s) | 1000 / 958 | 783.2 / 769.4 |
| prose decode, single stream (tok/s) | 33.8 at MTP k=2; 27.7 at k=4 (shipped) | 19.78 at k=4 |
| KV pool / max context | 803,968 tokens / 360K | 730,750 tokens / 360K |
| dense-layer quality gate (KL vs BF16) | PASS, 0.0284 | PASS, 0.0268 (thin margin) |

Numbers are single-request, one boot each, on our cluster, measured 2026-09-26 to 2026-09-30. Read
[RESULTS.md](RESULTS.md) for every step, the benchmark definitions and the caveats before comparing them with anything.

[kindling.json](kindling.json) has these numbers in a form a program can read; the Kindling AI site shows this
recipe from it. A change to this table changes kindling.json too. [AGENTS.md](AGENTS.md) has the rules.

## Hardware

- DGX Spark (GB10, 128 GB unified memory, one GPU per node), 6 or 4 nodes.
- Each node's ConnectX-7 on a 200G switch (we used a MikroTik CRS812), RoCEv2, MTU 1500. NCCL uses both RoCE
  HCAs of the one cabled port (`NCCL_IB_HCA=<hca0>,<hca1>`, `NCCL_IB_MERGE_NICS=1`).
- One model per cluster: nothing else should hold GPU memory on these nodes.

## What each layer of the stack does

Both builds start from vLLM on the MiaAI-Lab ARM image with the r17 EXL3 "TR3" loader and b12x kernels
(`image/`), then bind-mount these overlays at launch (`overlays/`):

| layer | files | what it does | TP6 | TP4 |
|---|---|---|---|---|
| expert re-fragmenting | `runtime/tp6-fragments/` | re-exports the TP4 checkpoint for six ranks without requantizing: each expert's four original 512-channel fragments go to different ranks (`(4·expert + part + layer) % 6`), attention padded 64 → 72 heads | yes | – |
| E3 prefill | `third_party/e3/` (AGPL-3.0) | grouped direct-trellis MoE GEMM for prefill (MiaAI-Lab's E3, extended for K3/K4 experts and our fragment layouts) | yes | yes |
| D2d FP8 dense | `overlays/dense-fp8-gptq/d2_fp8.py` + `overlays/vllm-tp*/exl3.py` | all BF16 non-expert linears + lm_head become MXFP8: Marlin W8A16 for small batches, FlashInfer MXFP8 W8A8 GEMM for ≥ 256 rows (checked at load, falls back to dequant + BF16) | yes | yes |
| D8 GPTQ int4 decode | `overlays/dense-fp8-gptq/d2_fp8_gptq.py` + per-rank codes (`gptq/`) | for batches < 256 rows, 312 dense modules per rank run int4 g32 GPTQ through Marlin; +~1.9 GB/rank; `VLLM_MARLIN_USE_ATOMIC_ADD=1` on top | yes | yes |
| MTP fix | `overlays/vllm-tp*/deepseek_mtp.py` | MTP `eh_proj` as a replicated linear so D2 can convert it | yes | yes |
| D3 RoCE all-reduce | `overlays/d3-roce/` | b12x one-shot RoCE all-reduce (≤ 256 KB) and all-gather (≤ 2 MB) replacing NCCL for small messages | yes | yes |
| DCP2 + draft slots | `overlays/vllm-tp4/{sm120_dcp,mla_attn_dcp,spec_decode_utils,llm_base_proposer}.py` | decode context parallel 2 on the SM120 sparse-MLA backend, with DCP-aware MTP draft KV slots | – | yes |
| fast loading | `overlays/tp6-fastload/`, `overlays/tp4-load-accel/` | O_DIRECT thread-pool expert loading and a filter that skips weights the rank discards; TP4 also caches prepared expert layouts on disk | yes | yes |
| graph warmup | `overlays/vllm-common/` | FULL CUDA-graph warmup including the EXL3 route-pack kernels | yes | yes |

## Reproduce

1. **Image.** `image/build-context.sh tp4|tp6 <r17 b12x package dir> <context dir>`, then `docker build` there
   (TP6 is three Dockerfiles in order: `Dockerfile.1-fragments` → `.2-fastload` → `.3-e3`, tag each as the next
   one's `FROM`). The r17 b12x source is not vendored here; [image/README.md](image/README.md) says how to get it
   from the base commit and the release patch. The base image is
   `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`.
2. **Weights.** Download the TR3 3.25 bpw checkpoint to every node. TP4 uses it as is. TP6: on each node run
   `runtime/tp6-fragments/export_rank.py --source <ckpt> --output <rank dir> --rank <r>` (stdlib only, byte-copies
   only that rank's fragments; it expects a `MANIFEST.sha256` of the source files: the checkpoint ships one next to
   the weights, so download it with them).
3. **GPTQ codes.** Download our solved per-rank codes from Hugging Face,
   [adapt-ai-systems/glm-5.3-full-exl3-gptq-dense](https://huggingface.co/adapt-ai-systems/glm-5.3-full-exl3-gptq-dense)
   (`tp6/r<rank>/` or `tp4/r<rank>/`), or solve your own per rank with `gptq/` (a capture boot, then a CPU solve;
   see `gptq/README.md`). Our calibration text is private, so codes you solve will differ from ours. Either way,
   run the KL gate (`eval/README.md`).
4. **Launch.** Copy `launch/tp6/launch.env.example` or `launch/tp4/launch.env.example`, fill in addresses and paths,
   check the printed command (`DRY_RUN=1`, the default), then `DRY_RUN=0` on each worker and last on the head.
   Wait for `/health`, then send a smoke request.
5. **Check.** `eval/pbench.py --effort low` should land near the RESULTS.md rows. On TP4, if the first 8K prefill
   runs below ~580 tok/s, reboot once (see RESULTS.md).

## Optional on TP6: adaptive draft length (D13, phase mode)

D13 replaces the fixed MTP draft length (k=4) with a scheduler that picks k each decode step from 1 to 4,
using a running estimate of draft acceptance and a measured step-cost table. In `phase` mode it keeps separate
acceptance state for reasoning, content, code fences and tool calls. Code and details:
[`experiments/d13-adaptive-k/`](experiments/d13-adaptive-k/README.md).

**Enable:** set `D13=1` for `launch/tp6/launch.sh` on every rank (off by default). It binds
`experiments/d13-adaptive-k/adaptive_k_phase.py` and the three files in `experiments/d13-adaptive-k/overlay/`, adds
`--scheduler-cls vllm.v1.core.sched.adaptive_k_phase.AdaptiveKScheduler`, sets the capture sizes to
`[2,3,4,5,6,8,9,10,12,15,16,20]` and the env `GLM_D13_QLENS=2,3,4,5 GLM_D13_NO_CG_ROUND=1 VLLM_ADAPTIVE_K_MODE=phase
VLLM_ADAPTIVE_K_COST_MS=2:55,3:62,4:70,5:77`.

**Measured** (TP6, one window, A and D alternating: D, A, D, A; n = boots per build; tok/s; GPU clocks locked to
≤ 2200 MHz with `nvidia-smi -lgc 0,2200`, so not comparable with the full-clock rows above). Both builds ran the
same container config (our measured TP6 stack, with `ATOMIC=0` and `tools/kring` loaded); the only difference is D13.

| clk2200 | A: fixed k=4 | D: D13 phase | D vs A |
|---|---|---|---|
| prefill 8K | 1038.4 (n=2) | 1045.7 (n=2) | +0.7% |
| prefill 32K | 999.4 (n=2) | 998.6 (n=2) | -0.1% |
| prose decode (`eval/pbench.py --effort low`) | 28.7 (n=2) | 32.1 (n=2) | +11.9% |
| real single-turn prompts (10 prompts, T=0.7, 700 tok; prompts not published) | 34.7 (n=2) | 36.3 (n=2) | +4.7% |
| structured output (MTP acceptance bench, T=0) | 61.3 (n=2) | 60.7 (n=2) | -1.0% |
| code (same bench) | 47.2 (n=2) | 46.8 (n=2) | -0.7% |
| 2 concurrent sessions, aggregate | 50.2 (n=2) | 47.9 (n=2) | -4.5% |
| 4 concurrent sessions, aggregate | 75.1 (n=2) | 79.4 (n=2) | +5.8% |
| KV pool (tokens) | 803,968 | 803,968 | same |

D gains on prose and real prompts (both D runs above both A runs on each of those rows); structured output and
code are about 1% slower; prefill and KV are unchanged. D leaves ~0.4 GB less free memory on the head node at idle.
A 455 s mixed-load soak on D (3 concurrent clients: cold 8K-48K prompts, real prompts, tool/JSON/code) finished
with 0 errors and 0 stalls (no A soak for comparison).

**Hangs.** D hung in 2 of its 20 boots on our cluster, both early and at full clock: once during boot (a
shared-memory broadcast stall, while our network was having an outage) and once on the first cold 8K prefill after
a short request, the same trigger as the TP6 prefill hang under Caveats. Two more of the 20 boots came before the 16
below and are in neither count. The 16 D boots since then, 15 of them with clocks locked to ≤ 2200 MHz, all came up
on the first try and answered a short and a cold ~8K request (0 of 16 hung; that still allows a hang rate of up to
about 17% at 95% confidence). Check `/health` and send one short and one cold ~8K request after every boot before you
rely on D.

## Caveats

- **Not re-run from this repo.** The launchers are reconstructed from the recorded container configs of the measured
  runs (same flags, env and bind targets) and were only dry-run checked. The images were built step by step on our
  nodes and not rebuilt from `image/`. Expect to debug.
- **Known prefill hang on TP6** that we did not root-cause; on our cluster a CUPTI injection library
  made it go away; it is now in `tools/kring`. Details in RESULTS.md "Known problems".
- The GPTQ codes are calibration-dependent; the TP4 gate passed by 0.12 points of top-1.
- Measurements are single-stream except one concurrency table; long-context prefill was measured on TP6 only.
- Some vLLM files were taken from the base image; we treat them as vLLM's Apache-2.0 code.

## Layout

```
launch/        tp6/ and tp4/: launch.sh + launch.env.example
overlays/      files bind-mounted over the image at launch (see table above)
image/         Dockerfiles, build-context.sh, r17 EXL3 quantization module, upstream pins
runtime/       TP6 expert re-fragmenting export + loader adapters (baked into the TP6 image)
third_party/   E3 prefill runtime (AGPL-3.0)
gptq/          D8 calibration capture and GPTQ solve (TP6, TP4)
eval/          prefill/decode bench, KL gate, MTP acceptance
experiments/   adaptive draft length (opt-in on TP6 via D13=1, see above); measured but not kept: EXL3 fused MoE prefill kernel, TP4 profiles
kindling.json  the results above, machine-readable, for the Kindling AI site
```

## License

Our code is MIT ([LICENSE](LICENSE)). vLLM-derived files and b12x code are Apache-2.0; the E3 runtime is AGPL-3.0;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for each path and the people whose work this builds on.
