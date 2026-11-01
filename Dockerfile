# ARDY served over the pedestrian motion contract (gRPC, :50061).
#
#   docker run --gpus all -p 50061:50061 \
#     -v ~/.cache/huggingface:/hf -e HF_HOME=/hf \
#     ghcr.io/hakuturu583/carla_ardy_backend:<version>
#
# The Hugging Face cache is mounted, not baked in: the checkpoint and the gated
# Llama-3-8B text encoder are downloaded into it once, with the host's token.
FROM python:3.11-slim

# git: uv fetches ARDY from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # The downloaded wheels are not the image's: no cache left in a layer.
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # The MotionCorrection C++ extension corrects foot skating, which the
    # backend does not ask for: not built (no CMake, no compiler here).
    ARDY_SKIP_MOTION_CORRECTION=1

WORKDIR /app

# The dependencies first, so a change to the server alone reuses their layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra ardy --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra ardy

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/hf \
    # The text encoder runs in this process: there is no encoder service to
    # try first.
    TEXT_ENCODER_MODE=local \
    PORT=50061 \
    PYTHONUNBUFFERED=1

EXPOSE 50061
CMD ["carla-ardy-backend"]
