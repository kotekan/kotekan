# Kotekan / CHORD Build Environment

This repository contains the Docker build definition for the **Kotekan** and **CHORD** scientific signal processing pipeline.

It provides a **reproducible, high-stability environment** based on Ubuntu 24.04 (Noble Numbat), with strictly pinned compilers and scientific libraries to ensure bit-exact build reproducibility.

## 📋 Prerequisites (Do this once)

Before building the project, you must ensure your **Host Machine** is configured to allow Docker to access your GPU.

### 1. Install Docker Engine
(Run these commands inside your WSL/Linux terminal)
#### 1. Install Docker using the official convenience script
```bash
curl -fsSL https://get.docker.com | sudo sh
```
#### 2. Add your user to the "docker" group (allows running without 'sudo')
```bash
sudo usermod -aG docker $USER
```
#### 3. Apply the group membership changes immediately
```bash
newgrp docker
````

### 2\. Install NVIDIA Container Toolkit

Standard Docker does not see GPUs by default. You must install this "glue" layer.

#### 1. Add the NVIDIA GPG key and repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
#### 2. Install the toolkit
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
#### 3. Configure Docker to use the NVIDIA runtime
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```
#### 4. Restart Docker
```bash
sudo systemctl restart docker
```

### 3\. Verify Setup

Run this command to confirm your host is ready. If this fails, the Kotekan build will not work.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

*Success:* You see a table listing your GPU and Driver Version.

-----

## 🚀 Quick Start

### 1\. Build the Image

You must build the image from inside the docker configuration directory so it finds `requirements.txt`.

#### 1. Navigate to the build definition folder
```bash
cd kotekan/tools/docker/24.04
```
#### 2. Build the container (this takes ~15 minutes)
```bash
docker build -t kotekan-build:latest .
```

### 2\. Run the Development Shell

**CRITICAL:** You must navigate back to the **root** of the repository before running this command, or the source code mount will be empty.

#### 1. Go up 3 levels to the repository root
```bash
cd ../../..
```
#### 2. Verify you see 'CMakeLists.txt'
```bash
ls -F 
```
#### 3. Run the container (Mounting the current directory to /code/kotekan)
```bash
docker run --rm -it \
    --gpus all \
    --user $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $(pwd):/code/kotekan \
    kotekan-build:latest
```

### 3\. Compile the Code

Once inside the container (`$USER@container:/code/build$`), run:

#### Configure and Build
```bash
cmake /code/kotekan
make -j$(nproc)
```

-----

## 🛠️ Critical Version Pins (The "Why")

This environment favors **Stability** over **Recency**. We have explicitly pinned specific versions to prevent "silent upgrades" from breaking the build pipeline.

| Component | Version | Justification |
| :--- | :--- | :--- |
| **OS** | Ubuntu 24.04 | Long Term Support (LTS) base. |
| **CUDA** | `13.0` | **Required by `gputils`.** Pinned to prevent `apt` from auto-upgrading to new/incompatible drivers. |
| **Compiler** | GCC `14` / Clang `18` | **Enables C++23.** GCC 14 is pinned to provide modern C++23 support for host-side code (CPU), aligning with CUDA 13 requirements. |
| **Python** | `3.12` | System default for Ubuntu 24.04. |
| **Numpy** | `1.26.4` | **CRITICAL:** Pinned \< 2.0. Numpy 2.0 breaks C-API binary compatibility for `kotekan` extensions. |
| **Julia** | `1.10.10` | **LTS Release**. Chosen over 1.11/1.12 to guarantee years of stability. |

## 📦 Managing Dependencies

### Python Packages (`requirements.txt`)

We do **not** install Python packages ad-hoc. All packages are defined in `requirements.txt`.

**To add/update a package:**

1. Edit `requirements.txt` (add your package, ideally with `==version`).
2. Rebuild the image: `docker build .`
3. **(Optional but Recommended)** Freeze the exact versions:
#### Run this inside the container to get the exact install list
```bash
pip freeze > requirements.lock
```

### Source Builds (Blaze, HighFive, etc.)

These libraries are built from source in the Dockerfile because `apt` versions are often too old or missing headers.

* **Location:** Source code is cloned into `/tmp/src`, built, installed to `/usr/local`, and then deleted to save space.
* **Updates:** To update `Blaze` or `HighFive`, edit the `git checkout` or `wget` tags in the **Source Builds** section of the `Dockerfile`.

-----

## ⚠️ Troubleshooting / FAQ

**Q: I get a "Symbol not found" error when importing `kotekan` in Python.**

> **A:** Check your Numpy version. Run `pip show numpy`. If it says `2.0.0` or higher, the C++ extensions are binary incompatible. Revert to `1.26.4`.

**Q: Why is the build so slow?**

> **A:** We build several C++ libraries (Blaze, HighFive, ASDF) from source. This is necessary for performance flags. Enable **Docker BuildKit** to speed this up:
> `DOCKER_BUILDKIT=1 docker build .`

**Q: Linter shows \
`$, '(', ')', ',', '.', '=', COMPOSITE_OPERATOR, IDENTIFIER, OTHER_PUNCT, QUOTE, SIMPLE_OPERATOR, '[', ']', '{' or '}' expected, got '}'` \
syntax errors in the Dockerfile.**

> **A:** This is a false positive thrown by generic linters. Ensure your editor is set to **Docker** language mode. The syntax `${VAR}` is valid Dockerfile syntax, even if some strict linters dislike the closing brace `}`.

-----

## 📂 Directory Structure

* `/code/build` - The default working directory (Build Artifacts).
* `/code/kotekan` - Intended mount point for your source code.
* `/usr/local/` - Installation prefix for Blaze, ASDF, HighFive, and Julia.
* `/tmp/src` - Temporary build folder (cleared after build).

**Note on Build Directory Separation:**
We deliberately build in `/code/build` (inside the container) rather than `/code/kotekan/build` (mounted from host).

* **Performance:** Compiling inside the container's native filesystem is significantly faster than compiling across a Docker volume mount (especially on macOS/Windows).
* **Hygiene:** This prevents thousands of temporary object files (`.o`) from polluting your local source tree or causing file permission issues on your host machine.

<!-- end list -->

-----