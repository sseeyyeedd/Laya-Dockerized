# Override for filtered networks, e.g.:
#   docker build --build-arg BASE_IMAGE=docker-mirror.liara.ir/library/python:3.12-slim .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV USE_TF=0

# Point pip at the Liara PyPI mirror
ENV PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple/
ENV PIP_TRUSTED_HOST=package-mirror.liara.ir
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

# The PyTorch CPU wheel (~196 MB) is vendored in docker-wheels/ because the
# download.pytorch.org CDN stalls inside the container network on this host.
# Refresh it with: curl -o docker-wheels/<wheel> <pytorch cpu index url>
COPY docker-wheels/torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl /wheels/

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir /wheels/torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY scripts/ ./scripts/

# Bake the staged local checkpoint (~645 MB) into the image so neither the
# build nor the runtime touches the HF hub. Refresh the staging dir with:
#   python scripts/stage_model.py
ENV HF_HOME=/model
COPY docker-model/ /model/hub/
RUN test -f /model/hub/models--convaiinnovations--laya/refs/main \
    || (echo "docker-model/ is missing or incomplete - run: python scripts/stage_model.py" && exit 1)

# Weights are baked in - forbid any hub access at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]