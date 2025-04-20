# VLM in robotics project for MIPT&Yandex bootcamp

## Installation

### Install ffmpeg
```
sudo apt update
sudo apt install ffmpeg
```

### Initialize SimplerEnv:

```git submodule update --init --recursive simpler_env```

### Initialize OpenVLA:

```git submodule update --init openvla```

### Download dependencies:

```conda env create -f environment.yml```

then use [hack](https://github.com/simpler-env/SimplerEnv/issues/26) to fully install SimplerEnv and OpenVLA:

```
conda activate vlm-in-robotics
pip install flash-attn==2.6.1
pip install tensorflow==2.15.0
pip install -r simpler_env/requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1
```

## Updating

### Update SimplerEnv:

```git submodule update --remote --recursive simpler_env```

### Update OpenVLA:

```git submodule update --remote openvla```

### Update dependencies:

```conda env update --file environment.yml --prune```

then use [hack](https://github.com/simpler-env/SimplerEnv/issues/26) to fully update SimplerEnv and OpenVLA:

```
conda activate vlm-in-robotics
pip install flash-attn==2.6.1
pip install tensorflow==2.15.0 # Update me
pip install -r simpler_env/requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # Update me
```

## Test

Install rt_1_x weights:

```./scripts/install_rt_1_x_checkpoints.sh```

Run test experiment:

```./experiments/000_stack_cube_rt1_x.sh```
