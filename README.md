# VLM in robotics project for MIPT&Yandex bootcamp

## Installation

### Initialize submodules:

```git submodule update --init --recursive simpler_env```
```git submodule update --init --recursive simpler_env_openvla```

### Download dependencies:

```conda env create -f environment.yml```

then use [hack](https://github.com/simpler-env/SimplerEnv/issues/26) to fully install SimplerEnv:

```
conda activate vlm-in-robotics
pip install tensorflow==2.15.0 # Update me
pip install -r simpler_env/requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # Update me
conda install ffmpeg
```

## Modules

### SimplerEnv

Test simpler env:

```bash
cd simpler_env
mkdir checkpoints
pip install gsutil
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
mv rt_1_x_tf_trained_for_002272480_step checkpoints
rm rt_1_x_tf_trained_for_002272480_step.zip
python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
--ckpt-path ./checkpoints/rt_1_x_tf_trained_for_002272480_step  --task widowx_stack_cube  --logging-root ./results_simple_eval/  --n-trajs 10
```

Result will be in results_simple_eval

## Server

```angular2html
cd server
docker compose up -d # Select only desired models in docker-compose.yml
```

# Notebooks

To run openvla_experiments and spatialvla_experiments you need to install simpler_env_openvla instead of common simpler_end
