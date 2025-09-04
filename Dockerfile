# Use NVIDIA CUDA base image with PyTorch support
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3-pip \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create output directory
RUN mkdir -p /app/outputs

# Set permissions
RUN chmod +x /app/main.py

# Expose port for any future web interface
EXPOSE 8000

# Create a simple startup script
RUN echo '#!/bin/bash\n\
echo "Story2Board Docker Container Started"\n\
echo "Available GPU devices:"\n\
nvidia-smi\n\
echo ""\n\
echo "To run Story2Board, use:"\n\
echo "python main.py --subject \"your subject\" --ref_panel_prompt \"reference prompt\" --panel_prompts \"panel1\" \"panel2\" --output_dir /app/outputs"\n\
echo ""\n\
echo "Example:"\n\
echo "python main.py --subject \"fox with shimmering fur\" --ref_panel_prompt \"stepping onto a mossy path\" --panel_prompts \"bounding across a fallen tree\" --output_dir /app/outputs"\n\
echo ""\n\
echo "Container is ready for commands..."\n\
/bin/bash' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]