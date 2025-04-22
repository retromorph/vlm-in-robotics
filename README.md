# VLM in robotics project for MIPT&Yandex bootcamp

## Installation

### Initialize SimplerEnv:

```git submodule update --init --recursive simpler_env```

### Download dependencies:

```conda env create -f environment.yml```

then use [hack](https://github.com/simpler-env/SimplerEnv/issues/26) to fully install SimplerEnv:

```
conda activate vlm-in-robotics
pip install tensorflow==2.15.0 # Update me
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # Update me
```

## Updating

### Update SimplerEnv:

```git submodule update --remote --recursive simpler_env```

### Update dependencies:

```conda env update --file environment.yml --prune```

then use [hack](https://github.com/simpler-env/SimplerEnv/issues/26) to fully update SimplerEnv:

```
conda activate vlm-in-robotics
pip install tensorflow==2.15.0 # Update me
pip install -r simpler_env/requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # Update me
```

## Modules

### [Llserver](https://github.com/AmpiroMax/llserver)
API gateway for LLMs

### SimplerEnv

Test simpler env:

```bash
python simpler_env/simple_inference_visual_matching_prepackaged_envs.py --policy rt1 \
--ckpt-path ./checkpoints/rt_1_x_tf_trained_for_002272480_step  --task widowx_stack_cube  --logging-root ./results_simple_eval/  --n-trajs 10
```
