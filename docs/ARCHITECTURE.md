# Architecture

NPU Harness is a single FastAPI process. It does three jobs: it watches the NPU, manages models through one
or more inference engines, and puts an OpenAI-compatible API in front of those engines.

```
                 browser (dashboard)                any OpenAI client
                        │                                   │
                        ▼                                   ▼
┌───────────────────────────────── npu-harness (FastAPI) ─────────────────────────────────┐
│  /api/npu*        /api/models, /api/jobs          /v1/*                                  │
│      │                   │                          │                                    │
│  NpuMonitor         ModelRouter ─────────┐     InferenceProxy ──── request stats         │
│  (thread)            │        fallback   │       │         │                             │
│   D3DKMT + PDH       ▼                   ▼       ▼         ▼                             │
│                FoundryLocalBackend   OpenVinoBackend (in-process LLMPipeline)            │
└──────────────────────│─────────────────────────────────────│────────────────────────────┘
                       │ `foundry -o json …` (CLI → IPC)     │ OpenVINO GenAI → NPU driver
                       ▼                                     ▼
              Foundry Local daemon ── /v1 (HTTP, random port)     NPU
                       │
            ONNX Runtime + execution providers (OpenVINO EP, QNN, …) → NPU / GPU / CPU
```

## Source layout

```
src/npu_harness/
  __main__.py          CLI entry point (`npu-harness`)
  app.py               FastAPI app: routes, wiring
  telemetry/
    d3dkmt.py          adapter discovery via gdi32 D3DKMT* (finds the NPU and its LUID)
    pdh.py             minimal ctypes wrapper over Windows performance counters (PDH)
    monitor.py         background sampler: utilization, memory, per-process usage, history
  backends/
    base.py            Backend protocol, ModelVariant, BackendError
    foundry.py         Foundry Local, driven through its CLI's JSON output
    openvino.py        OpenVINO GenAI, in-process, for Foundry's OpenVINO model builds
    router.py          Foundry first, OpenVINO fallback; remembers what's broken
  jobs.py              background jobs (download / load / unload / remove)
  proxy.py             /v1 proxy, in-process routing, per-request stats
  static/              dashboard: index.html, styles.css, app.js (no build step)
scripts/
  capture_screenshots.py   regenerates docs/screenshots
```

## Telemetry

**Finding the NPU.** Windows runs NPUs under the compute-only driver model (MCDM): to the OS an NPU is a
graphics adapter that can only do compute. `D3DKMTEnumAdapters2` skips these, so `d3dkmt.py` calls
`D3DKMTEnumAdapters3` with `IncludeComputeOnly` set, then `D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERTYPE)` on
each adapter. The NPU is the adapter that is compute-only, not a software device, and has no display.
Its **LUID** is the key into the performance counters.

The friendly name, vendor and driver version come from `Win32_PnPSignedDriver` (device class
`COMPUTEACCELERATOR`). That WMI query takes a few seconds, so it runs on a side thread and the dashboard fills
it in when it arrives.

**Sampling.** `NpuMonitor` samples once a second on a background thread, using these counters:

| Counter | Used for |
|---|---|
| `\GPU Engine(*)\Utilization Percentage` | Utilization. Instances look like `pid_123_luid_0x…_phys_0_eng_0_engtype_Compute`. For each engine on the NPU's LUID, usage is summed across processes; the busiest engine is the device's utilization, which is how Task Manager computes it. |
| `\GPU Adapter Memory(*)\Shared Usage`, `Dedicated Usage` | NPU memory. NPUs use shared system RAM. |
| `\GPU Process Memory(*)\Shared Usage` | Per-process NPU memory |

Process names come from `psutil`. The last 5 minutes are kept in memory and streamed to the browser as
server-sent events (`/api/npu/stream`).

**What isn't available.** No public Windows or vendor API exposes NPU temperature, power or clock speed.

## Backends

All backends speak the protocol in `backends/base.py`: catalog, download, load, unload, remove. Errors are
`BackendError`s carrying a message, code, hint and detail, which the UI shows as-is.

### Foundry Local

Foundry Local 2.x serves only an OpenAI-compatible `/v1` API over HTTP. Model management goes through the
`foundry` CLI, which talks to the same daemon over IPC. The backend runs `foundry -o json <command>` and parses
the result. Results arrive on stdout; errors arrive as `{"error": {...}}` on **stderr**.

Going through the daemon means anything loaded by the harness is also usable by every other client of that
daemon.

When a load fails with Foundry's generic `INTERNAL_ERROR`, the backend reads `foundry server logs` and
pattern-matches the real cause (e.g. `Error creating execution provider: …`) into the error's detail and hint.

Safety note: `foundry cache rm` with no model argument clears the **whole** cache, so `remove()` refuses an
empty id.

### OpenVINO GenAI

Foundry's `*-openvino-npu` and `*-openvino-gpu` variants are OpenVINO IR (`openvino_model_dy.xml/.bin`,
INT4) plus OpenVINO tokenizers, wrapped in a tiny ONNX file with an `EPContext` node that points ONNX
Runtime's OpenVINO execution provider at the IR. OpenVINO GenAI can load the IR directly, without ONNX Runtime
or the Windows execution-provider package.

To load a model, the backend:

1. **Finds** the cached folder: every Foundry model folder holds an `inference_model.json` whose `Name` is the
   variant id.
2. **Stages** it under `%LOCALAPPDATA%\NPU-Harness\openvino\models\<id>`, hard-linking files into the names
   `LLMPipeline` expects (`openvino_model.xml/.bin`, `openvino_tokenizer.*`, `openvino_detokenizer.*`, configs).
   Hard links cost no disk space and leave Foundry's cache untouched; it falls back to copying across volumes.
3. **Applies Foundry's settings** from the model's `genai_config.json`: the NPU compile properties (e.g.
   `MAX_PROMPT_LEN`, `MIN_RESPONSE_LEN`) and the sampling defaults (`temperature`, `top_p`, `top_k`,
   `repetition_penalty`).
4. **Compiles** with `LLMPipeline(path, "NPU", CACHE_DIR=…)`. The compiled blob is cached: the first load of
   qwen2.5-0.5b took about 90 s on a Core Ultra 7 258V, and later loads take about 2–3 s.

Inference runs in a worker thread, one generation at a time per model. The messages are templated with the
model's chat template, and the result is returned as OpenAI `chat.completion` objects or SSE
`chat.completion.chunk` streams. OpenVINO's streamer emits word-sized chunks that can contain several tokens,
so exact token counts come from the generation's perf metrics, not from counting chunks. If the client
disconnects mid-stream, generation is cancelled at the next token.

### Router and fallback

`ModelRouter.load()` always tries Foundry first. If Foundry fails on an OpenVINO NPU/GPU variant, the router
loads the same files with the OpenVINO backend. The job then succeeds with a **notice** explaining what
happened, which the UI shows as a yellow banner. The failure is remembered per execution provider for the rest
of the session, so later loads go straight to OpenVINO. If both engines fail, the error carries both causes.

## Inference proxy

`/v1/*` is the harness's stable OpenAI endpoint. Foundry picks a new port each time its daemon starts; the
proxy rediscovers it, retrying once on connection errors.

- **Chat requests for in-process models** (`chat/completions` whose `model` is loaded in OpenVINO) are
  answered by `OpenVinoBackend.chat()`.
- **Everything else** is streamed through to Foundry unchanged.

Both paths record a `RequestStat`: time to first content token, duration, completion tokens and tokens/sec.
The last 100 are served at `/api/requests` for the dashboard. The clock starts before the request is sent,
because non-streaming responses only arrive once generation has finished.

## Jobs

Downloads, loads, unloads and removes can take minutes, so `JobManager` runs them as asyncio tasks. The
endpoint returns at once with a job id, and the UI polls `/api/jobs` once a second while anything is running.
Submitting the same action for the same model while one is already running returns the existing job. A job's
result may carry a notice (used by the fallback) alongside its message.

## Frontend

Plain HTML, CSS and JavaScript, with no framework and no build step. Charts are hand-rolled SVG with a
crosshair tooltip. Colors are CSS custom properties with a dark-mode set under `prefers-color-scheme`. All
text from outside (model names, process names, error messages) is inserted with `textContent`, never
`innerHTML`.
