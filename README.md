# VLM in robotics project for MIPT&Yandex bootcamp

## Installation

Install ffmpeg
```
sudo apt update
sudo apt install ffmpeg
```

Initialize SimplerEnv:

```git submodule update --init --recursive simpler_env```

Initialize OpenVLA:

```git submodule update --init openvla```

Environment:

```conda env create -f environment.yml```

## Updating

Initialize SimplerEnv:

```git submodule update --remote --recursive simpler_env```

Initialize OpenVLA:

```git submodule update --remote openvla```

Environment:

```conda env update --file environment.yml --prune```
