# Use NVIDIA PyTorch base image with CUDA support for high-performance computing
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3
FROM $BASE_IMAGE

# Accept arguments for various settings
ARG LOCAL_DIR
ARG TENSORBOARD_PORT
ARG SERVICE_PORT

# Set environment variables for better Python behavior and capture build arguments
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV LOCAL_DIR ${LOCAL_DIR}
ENV TENSORBOARD_PORT ${TENSORBOARD_PORT}
ENV SERVICE_PORT ${SERVICE_PORT}

# Set the working directory
WORKDIR $LOCAL_DIR

# Install production dependencies
COPY requirements.txt $LOCAL_DIR/
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies
COPY requirements-dev.txt $LOCAL_DIR/
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the entrypoint script and set it as the entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Expose required ports
EXPOSE $TENSORBOARD_PORT
EXPOSE $SERVICE_PORT

# Default command to run when the container starts
CMD ["bash"]
