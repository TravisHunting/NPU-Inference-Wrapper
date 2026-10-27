# NPU Harness

A local dashboard and model manager for the NPU in your Windows PC, built on
[Foundry Local](https://github.com/microsoft/Foundry-Local).

- **Dashboard**: live NPU utilization, NPU memory, which processes are using the NPU, driver details.
- **Models**: browse Foundry Local's catalog filtered to NPU builds; download, load, unload and delete them.
  When a load fails, the harness digs the real cause out of Foundry's logs and explains it.
- **Playground**: chat with a loaded model and see time to first token and tokens/sec.
- **API**: a stable OpenAI-compatible endpoint at `http://127.0.0.1:8765/v1` (Foundry's own port changes on
  every restart), with per-request stats on the dashboard.

## Requirements

- Windows 11 with an NPU (Intel Core Ultra, AMD Ryzen AI, or Snapdragon X)
- [uv](https://docs.astral.sh/uv/) (or Python 3.12+)
- Foundry Local: `winget install Microsoft.FoundryLocal`

## Run

```bash
uv run npu-harness --open
```

Then open http://127.0.0.1:8765. The server binds to localhost only; pass `--host` to change that.

## How it works

| Piece | Source |
|---|---|
| Finding the NPU | `D3DKMTEnumAdapters3` with compute-only adapters included; the NPU is the hardware, compute-only (MCDM) adapter. Its LUID keys the performance counters. |
| Utilization and memory | Windows performance counters (`GPU Engine`, `GPU Adapter Memory`, `GPU Process Memory`), the same source Task Manager uses. No admin rights needed. |
| Driver details | `Win32_PnPSignedDriver` for the `ComputeAccelerator` device class. |
| Model management | The `foundry` CLI with `-o json`. Foundry Local 2.x has no management REST API, and going through the CLI keeps models in the shared daemon where any client can use them. |
| Inference | Proxied to the Foundry daemon's OpenAI-compatible `/v1` API. |

**What Windows doesn't expose:** NPU temperature, power draw and clock speed. Vendors don't publish them, so the
dashboard can't show them.

## Layout

```
src/npu_harness/
  telemetry/   d3dkmt.py (adapter discovery), pdh.py (perf counters), monitor.py (background sampler)
  backends/    base.py (backend interface), foundry.py (Foundry Local)
  jobs.py      long-running download/load jobs
  proxy.py     /v1 proxy + request stats
  app.py       FastAPI routes
  static/      the dashboard (plain HTML/CSS/JS, no build step)
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's next.
