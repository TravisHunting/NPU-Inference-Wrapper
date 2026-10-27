# Roadmap

## Goal

The easiest way for a developer to see what their PC's NPU can do and use it: what hardware they have,
what it's doing, which models run on it, and an API to call those models.

## Done (v0.1)

- [x] NPU discovery via D3DKMT (exact adapter LUID, not guesswork)
- [x] Live telemetry: utilization, shared memory, per-process usage, system RAM headroom
- [x] Foundry Local status, versions and the hardware it detects
- [x] Model catalog with NPU/GPU/CPU filter; download / load / unload / delete as tracked jobs
- [x] Load failures explained from the daemon log (e.g. execution-provider version mismatch)
- [x] Chat playground with streaming, TTFT and tokens/sec
- [x] Stable OpenAI-compatible `/v1` proxy with per-request stats

## Next

### Make the NPU path work on more machines
- [ ] **Execution-provider health check**: list installed Windows ML EP packages (`Get-AppxPackage *EP*`) with
      versions next to Foundry's ONNX Runtime version, and flag known mismatches before a load fails.
- [ ] Show download progress (the CLI's JSON mode only reports completion; parse text-mode progress or watch the cache directory).
- [ ] Detect when Foundry Local isn't installed and offer a one-click `winget install`.
- [ ] Test on AMD Ryzen AI and Snapdragon X machines; record which catalog models have NPU builds for each.

### Telemetry
- [ ] Mark inference requests on the utilization chart so you can see the NPU working for each request.
- [ ] Persist history beyond 5 minutes (SQLite), with a time-range selector.
- [ ] Per-model memory footprint after load (diff of process memory before and after).

### Models and inference
- [ ] Benchmark view: the same prompt on the NPU, GPU and CPU builds of one model; compare TTFT, tokens/sec and memory.
- [ ] Model info drawer: context length, tool-calling support, license (`foundry model info`).
- [ ] Register custom ONNX Runtime GenAI models (`foundry model register`).
- [ ] Embeddings and speech (`transcribe`) in the playground.

### Packaging
- [ ] Single-file Windows build (PyInstaller) and a tray icon that starts the harness at login.
- [ ] Optional API key when binding to a non-localhost interface.

### Other backends (behind `backends/base.py`)
- [ ] OpenVINO GenAI directly, for Intel models Foundry doesn't ship
- [ ] AMD Lemonade Server
- [ ] Windows ML for non-LLM models (vision, audio)

## Known issues

- **Intel NPU/GPU loads fail on this dev machine** (Foundry Local 0.12.1, ONNX Runtime 1.31.0, Windows ML OpenVINO
  EP package 1.8.95): `Experimental function OrtEpApi_ReleaseEpContextConfig_SinceV28 is not available in this build`.
  The OpenVINO EP package and Foundry's ONNX Runtime are out of step. CPU and WebGPU builds work. The harness explains
  this when it happens; the fix has to come from a Windows ML EP or Foundry Local update.
