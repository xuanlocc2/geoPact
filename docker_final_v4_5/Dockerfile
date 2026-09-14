FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS runtime-deps

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

FROM runtime-deps AS final

COPY app/dataset.py /app/dataset.py
COPY app/pet_physics.py /app/pet_physics.py
COPY app/pet_supervision.py /app/pet_supervision.py
COPY app/transforms.py /app/transforms.py
COPY app/unet.py /app/unet.py
COPY app/predict.py /app/predict.py
COPY app/entrypoint.py /app/entrypoint.py
COPY app/validate_checkpoint.py /app/validate_checkpoint.py
COPY app/validate_output.py /app/validate_output.py
COPY checkpoints/model.pth /app/checkpoints/model.pth

# Fail during build if the selected checkpoint is absent, incompatible, or
# belongs to a different input/model recipe.
RUN python /app/validate_checkpoint.py /app/checkpoints/model.pth

ENTRYPOINT ["python", "-u", "/app/entrypoint.py"]
