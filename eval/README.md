# Benchmarks and quality gate

All scripts talk to the vLLM OpenAI server (`--url`, default `http://localhost:8000`) and need only the Python stdlib.

- `pbench.py` (TP6) / `pbench_tp4.py` (adds `--no-decode` for a quick prefill probe): cold 8K/32K prefill,
  then 3 × 512-token prose decodes at T=0.7 with `reasoning_effort=low`, timing content tokens only.
  `--effort low` is how every number in RESULTS.md was taken. Prints one JSON summary line.
- `d8_kl.py`: the KL gate. Boot a BF16-dense reference (`GLM_D2_FP8=0`), `gen` (or use `kl-seqs.jsonl`, our
  16 × 256-token set) and `score` it; then boot the FP8 arm A (`GLM_D8_W4=0`) and the int4 arm
  (`GLM_D8_W4=1`), both with `GLM_D2_FP8_DEQUANT_M=1000000` so every scored row takes the decode path, `score`
  each, and `compare REF.jsonl ARM.jsonl`. Pass rule we used: int4 KL ≤ 2 × arm A's KL and top-1 agreement
  drop < 1 point vs arm A. `kl-seqs.jsonl` came from a TP6 boot; for TP4 we generated a fresh set the same way.
- `accept.py`: per-position MTP acceptance and streamed decode speed per workload (structured, code, prose), from
  the server's `/metrics` counters. It drives a decode prompt harness (`HARNESS_DIR` with `decode.py` and its
  frozen prompts) adapted from Matt Mastracci's `dev/repro/decode.py` in
  [kindlingai/glm-5.3-flash-gx10](https://github.com/kindlingai/glm-5.3-flash-gx10); the harness is not included.
