# carla_ardy_backend

[ARDY](https://github.com/nv-tlabs/ardy) is NVIDIA's autoregressive diffusion model of human
motion. This image serves it over the pedestrian motion contract (gRPC) of
[autoware_carla_scenario](https://github.com/autowarefoundation/autoware_carla_scenario), so a
CARLA scenario can move a pedestrian as ARDY generates it (`SmartPedestrianControllerAction`).
The scenario says what the pedestrian does in text, waypoints and full-body keyframes. ARDY
generates the motion a couple of seconds at a time, ahead of the run.

```text
autoware_carla_scenario                      this image (GPU)
SmartPedestrianControllerAction  --gRPC-->   carla-ardy-backend -> ARDY (+ Llama-3-8B text encoder)
```

You don't normally start it yourself. autoware_carla_scenario's queue starts the container
before a run whose scenario needs it (`@need_prepare`), after checking that the host has
the GPU and host memory free. It stops the container after the batch. The image and its
settings are `backends.ardy` of the scenario config.

## Image

```text
ghcr.io/hakuturu583/carla_ardy_backend:<version>   # every release
ghcr.io/hakuturu583/carla_ardy_backend:latest
```

To run it by hand:

```bash
docker run --rm --gpus all -p 50061:50061 \
  -v ~/.cache/huggingface:/hf \
  ghcr.io/hakuturu583/carla_ardy_backend:latest
```

### Hugging Face access

The text encoder is built on the gated
[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
model. Your Hugging Face account needs access to it, and the token must be in the mounted
cache (`hf auth login` writes it to `~/.cache/huggingface/token`) or in `HF_TOKEN`.

The first start downloads about 20 GB of weights into the cache. Later starts reuse them.

### Resources

The text encoder alone takes about 14 GB of GPU memory, so plan for about 24 GB of GPU
memory for the whole backend.

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `50061` | Port the gRPC server listens on |
| `ARDY_MODEL` | `soma` | Checkpoint (`soma`: ARDY-SOMA-RP-30FPS-Horizon60) |
| `ARDY_DEVICE` | `cuda` | Torch device |
| `ARDY_ACCELERATION` | `none` | `none` or `compile` (torch.compile) |
| `ARDY_HISTORY_FRAMES` | one second | Frames of history each step is generated from |
| `ARDY_DIFFUSION_STEPS` | the model's | Denoising steps |

## The contract

Clients talk to the server over `proto/pedestrian_motion/v0/pedestrian_motion.proto`:

- **Negotiate**
- **OpenSession**: takes the text timeline, root waypoints and full-body keyframes.
- **Step**: takes the frame to keep up to, and returns the generated frames after it.
- **CloseSession**

The contract belongs to autoware_carla_scenario. Its `.proto` and generated modules are
copied here byte for byte (see `proto/README.md`), and CI checks the copies.

The two sides use different frames:

- The contract is right-handed, z up, metres, with headings counter-clockwise from +x.
- ARDY is y up.

`frames.py` converts between them.

## Development

```bash
uv sync                      # the server and its tests (no torch)
uv run pytest
uv sync --extra ardy         # the model too (CUDA 12.8 torch)
uv run carla-ardy-backend
```

## Releases

Releases work the way they do in autoware_carla_scenario:

1. A pull request carries a `bump major`, `bump minor` or `bump patch` label. CI refuses
   one that has none.
2. When it merges, the release workflow bumps `version` in `pyproject.toml`, re-locks, and
   commits `chore: bump version to X.Y.Z`.
3. It tags `vX.Y.Z` and creates the GitHub Release.
4. It pushes the image as `ghcr.io/hakuturu583/carla_ardy_backend:X.Y.Z` and `:latest`.
