# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:6.4.1
FROM ${BASE_IMAGE}

COPY . /rocm-ls-examples

WORKDIR /rocm-ls-examples
RUN ls -l /rocm-ls-examples
RUN ls -l requirements.txt
RUN ls -l supervisord.conf

# Set ROCm environment variables
ENV HIP_PATH=/opt/rocm
ENV PATH=$HIP_PATH/bin:$PATH
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=$HIP_PATH/lib:$LD_LIBRARY_PATH
ENV ROCM_HOME=/opt/rocm

# Install dependencies
RUN apt update && \
    apt install -y software-properties-common lsb-release gnupg && \
    apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc && \
    add-apt-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt update && \
    apt install -y git wget gcc g++ ninja-build git-lfs       \
                    yasm libopenslide-dev python3.10-venv        \
                    cmake rocjpeg rocjpeg-dev rocthrust-dev      \
                    hipcub hipblas hipblas-dev hipfft hipsparse  \
                    hiprand rocsolver rocrand-dev rocm-hip-sdk libvips supervisor


# Set up Python venv
WORKDIR /rocm-ls-examples
RUN python3.10 -m venv /venv

# Upgrade pip, setuptools, and wheel first
RUN /venv/bin/python -m pip install --upgrade pip setuptools wheel

# Install Python requirements from the copied folder
RUN /venv/bin/pip install -r requirements.txt

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set venv in PATH for runtime (after all build steps are complete)
ENV PATH="/venv/bin:$PATH"

# Expose Streamlit port
EXPOSE 8501

# Entrypoint: supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
