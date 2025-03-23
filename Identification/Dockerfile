# Use an official Ubuntu image as the base
FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Install dependencies for Python, Julia, and system tools
RUN apt-get update -y && \
    apt-get install -y gcc make wget zlib1g-dev libffi-dev libssl-dev curl tar python3 python3-pip

# Install Julia version 1.11.4 for ARM (Apple M1/M2)
RUN wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.11/julia-1.11.4-linux-aarch64.tar.gz && \
    tar -xvzf julia-1.11.4-linux-aarch64.tar.gz -C /opt && \
    ln -s /opt/julia-1.11.4/bin/julia /usr/local/bin/julia

# Verify Julia installation
RUN julia --version

# Install Python packages and Julia packages
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install julia

# Optionally: Install PyCall in Julia and configure it
RUN julia -e 'using Pkg; Pkg.add("PyCall")'

# Install the Julia-Python bridge (PyCall) for Julia from Python
RUN python3 -c 'import julia; julia.install()'

# Set environment variables for Python and Julia (optional)
ENV PYTHON=python3

# Run the container with your app (if any)
# CMD ["python", "your_app.py"]  # Example for running Python app (replace with your actual app)
