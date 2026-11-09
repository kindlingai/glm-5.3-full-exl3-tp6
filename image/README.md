# image/

Docker build inputs for the TP4 and TP6 images. `build-context.sh` assembles a build context from this folder, the
overlays and one thing that is not in this repo: the r17 b12x package. This page says how to get it.

## b12x

`build-context.sh tp4|tp6 <b12x dir> <context dir>` wants the `b12x/` Python package (the folder that holds
`__init__.py`) as its second argument. The repo does not vendor it. Build it from the pins in
`pins/b12x.lock.json`:

1. Clone [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) and check out the base commit
   `c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5`.
2. Get the r17 release patch `integration.patch` from
   [local-inference-lab/blackwell-llm-docker @ 64f5c35](https://github.com/local-inference-lab/blackwell-llm-docker/tree/64f5c3503ecb8d3d315340c8699a18266af9586c/patches/releases/infernal-invocation-r17/b12x).
   Check that its SHA256 is `9dde29254d7db866f189eea416c1c2de939d7e6e804d4d8f594368a701b12cf2`.
3. In the b12x clone, run `git apply /path/to/integration.patch`.
4. Pass the `b12x/` folder of that clone (the one with `__init__.py`) as the second argument:

```
image/build-context.sh tp6 /path/to/b12x-clone/b12x /path/to/context
```

Then `docker build` in the context folder, as the main [README](../README.md) says under "Reproduce".

`pins/vllm.lock.json` records the vLLM pin in the same way. The licences and credits for both are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
