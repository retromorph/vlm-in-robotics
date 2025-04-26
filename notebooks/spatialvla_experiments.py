#!/usr/bin/env python3
# ================================================================
#  Spatial-VLA evaluation with extra metrics + video logging
# ================================================================
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
import numpy as np
import mediapy as media
from pathlib import Path
from collections import defaultdict
import sapien.core as sapien
import math


######################
import torch, os
torch._dynamo.config.suppress_errors = True   # не падать, а тихо откатываться
torch._dynamo.disable()                       # полностью выключить граф-банкирование
# или переменная окружения
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCH_COMPILE"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


import transformers
import types

# в новых версиях осталась make_flat_list_of_images; используем её
from transformers.image_utils import make_flat_list_of_images

import transformers.models.paligemma.processing_paligemma as pp

if not hasattr(pp, "make_batched_images"):
    # создаём совместимую обёртку
    def make_batched_images(images):
        """
        Fallback-реализация: просто разворачивает вложенные списки
        в плоский и возвращает его (то же поведение).
        """
        return make_flat_list_of_images(images)

    pp.make_batched_images = make_batched_images
# ---------------------------------------------------------------

from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
os.environ["TORCH_COMPILE"] = "0"
os.environ["HF_HOME"] = "/cache/huggingface"
import numpy as np
import mediapy
######################


# -------- collision utilities  -----------------------------------
import sapien.core as sapien

def robot_links(scene):
    """Return the set of links that belong to the *first* SAPIEN articulation."""
    arts = scene.get_all_articulations()
    return set(arts[0].get_links()) if arts else set()

def count_robot_env_contacts(scene, links):
    """Contacts where exactly one body is a robot link."""
    # print(links)
    n = 0
    for cp in scene.get_contacts():          # list[Contact]
        a0, a1 = cp.actor0, cp.actor1
        in0, in1 = a0 in links, a1 in links
        if in0 ^ in1:                        # XOR → env-robot, not self-collision
            n += 1
            # print(a0, a1)
    return n
# ------------------------------------------------------------------

HAND_KEYWORDS = ("gripper", "finger", "tcp", "hand", "wrist")

def hand_links(scene):
    """Subset of robot links that a task can physically collide with."""
    rlinks = robot_links(scene)
    return {lk for lk in rlinks
            if any(k in lk.get_name().lower() for k in HAND_KEYWORDS)}


# --------------------------------------------------
TASKS       = ["widowx_put_eggplant_in_basket", "google_robot_pick_coke_can", "google_robot_open_drawer"]

N_EPISODES  = 30          # per task
FPS         = 10          # video fps
CKPT_PATH   = "spatialvla/spetialvla-4b"
VID_ROOT    = Path("videos/spatial_vla")          # videos/<task>/episode_XX_*.mp4
# --------------------------------------------------

# def get_tcp(env):
#     """Return TCP (tool-center-point) Cartesian position in world frame."""
#     return np.asarray(env.get_tcp_pose()[:3])     # (x,y,z)


## ------------------------------------------------------------------
#def tcp_position(obs, env):
#    """
#    Return tool-center-point (x,y,z) for *any* SIMPLER env.
#    Search order:
#      1) observation keys
#      2) env / inner wrappers that expose get_tcp_pose()
#      3) gripper_tcp link pose in SAPIEN scene
#    """
#    import numpy as np
#
#    # 1) ---- look inside the observation ---------------------------------
#    if isinstance(obs, dict):
#        for k in ("tcp_pose", "hand_pose"):
#            if k in obs:
#                return np.asarray(obs[k][:3])
#        if "agent" in obs and "tcp_pose" in obs["agent"]:
#            return np.asarray(obs["agent"]["tcp_pose"][:3])
#
#    # 2) ---- ask any wrapper that has get_tcp_pose -----------------------
#    try:
#        fn = env.get_wrapper_attr("get_tcp_pose")  # may raise AttributeError
#        if fn is not None:
#            return np.asarray(fn()[:3])
#    except AttributeError:
#        pass                                        # simply try the next method
#
#    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_tcp_pose"):
#        return np.asarray(env.unwrapped.get_tcp_pose()[:3])
#
#    # 3) ---- fall back to SAPIEN gripper_tcp link -------------------------
#    scene = getattr(env.unwrapped, "_scene", None) or getattr(env.unwrapped, "scene", None)
#    if scene is not None:
#        for art in scene.get_all_articulations():
#            for link in art.get_links():
#                if "gripper_tcp" in link.get_name():
#                    return np.asarray(link.get_pose().p)   # Pose → position
#
#    raise RuntimeError("Could not locate TCP position in obs or env.")
## ------------------------------------------------------------------
def tcp_position(obs, env):
        """
        Return tool-center-point (x,y,z) for *any* SIMPLER env.
        Search order:
        1) observation keys
        2) env / inner wrappers that expose get_tcp_pose()
        3) gripper_tcp link pose in SAPIEN scene
        """

        # 1) ---- look inside the observation ---------------------------------
        if isinstance(obs, dict):
            for k in ("tcp_pose", "hand_pose"):
                if k in obs:
                    return np.asarray(obs[k][:3])
            if "agent" in obs and "tcp_pose" in obs["agent"]:
                return np.asarray(obs["agent"]["tcp_pose"][:3])
            if "extra" in obs and "tcp_pose" in obs["extra"]:
                return np.asarray(obs["extra"]["tcp_pose"][:3])

        # 2) ---- ask any wrapper that has get_tcp_pose -----------------------
        try:
            fn = env.get_wrapper_attr("get_tcp_pose")  # may raise AttributeError
            if fn is not None:
                return np.asarray(fn()[:3])
        except AttributeError:
            pass                                        # simply try the next method

        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_tcp_pose"):
            return np.asarray(env.unwrapped.get_tcp_pose()[:3])

        # 3) ---- fall back to SAPIEN gripper_tcp link -------------------------
        scene = getattr(env.unwrapped, "_scene", None) or getattr(env.unwrapped, "scene", None)
        if scene is not None:
            for art in scene.get_all_articulations():
                for link in art.get_links():
                    if "gripper_tcp" in link.get_name():
                        return np.asarray(link.get_pose().p)   # Pose → position

        raise RuntimeError("Could not locate TCP position in obs or env.")



# --- metric containers -------------------------------------------------------
metrics = {t: defaultdict(list) for t in TASKS}   # per-task episode lists

for TASK in TASKS:
    # 1)  Build env & policy ---------------------------------------------------
    env = simpler_env.make(TASK)

    # choose policy_setup automatically
    setup = "google_robot" if TASK.startswith("google_robot") else "widowx_bridge"

    policy = SpatialVLAInference(
    saved_model_path="openvla/openvla-7b",
    policy_setup="google_robot",
)
    # SpatialVLAInference(saved_model_path=CKPT_PATH,
    #                              policy_setup=setup)

    # create output dir for this task
    outdir = VID_ROOT / TASK
    outdir.mkdir(parents=True, exist_ok=True)

    # 2)  Roll-out N_EPISODES --------------------------------------------------
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        instr  = env.get_language_instruction()
#         instr   = f'''Instruction: Open the middle drawer fast and fully .
# Environment Context:
#   - You are controlling a robotic system with multiple drawers labeled as "top," "middle," and "bottom."
#   - The drawers are part of a simulated or real-world environment designed for task-based interactions.
# Task Requirements:
#   - Generate a precise command or sequence of actions to open the middle drawer.
# Expected Output: A clear and executable instruction or code snippet to open the middle drawer.'''

        # instr = 'Pick a can of Coke from the table, please, as much as possible, neat and clear'


        policy.reset(instr)

        scene = env.unwrapped._scene if hasattr(env.unwrapped, "_scene") else env.unwrapped.scene
        rlinks = hand_links(scene)
        prev_contact_count = 0

        frames = []
        success = trunc = False

        steps, path_len, collision_events = 0, 0.0, 0
        prev_tcp = tcp_position(obs, env)

        while not (success or trunc):
            img = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(img)

            _, act = policy.step(img, instr)
            vec    = np.concatenate([act["world_vector"],
                                     act["rot_axangle"],
                                     act["gripper"]])
            obs, _, success, trunc, info = env.step(vec)

            # -------- metric accumulation ------------------------------------
            steps += 1

            # Calculate path length

            tcp = tcp_position(obs, env)
            path_len += np.linalg.norm(tcp - prev_tcp)
            prev_tcp = tcp

            # cache once
            if "scene" not in locals():
                scene = env.unwrapped._scene if hasattr(env.unwrapped, "_scene") else env.unwrapped.scene
                rlinks = hand_links(scene)

            # compute current number of environment–robot contacts
            curr_contact_count = count_robot_env_contacts(scene, rlinks)
            # only count an “event” if it actually changed since last frame
            if curr_contact_count > prev_contact_count:
                collision_events += 1
            prev_contact_count = curr_contact_count
            # print(f"contacts: {curr_contact_count}, events: {collision_events}")
            # -----------------------------------------------------------------

        # save video ----------------------------------------------------------
        tag = "success" if success else "fail"
        video_path = outdir / f"episode_{ep:02d}_{tag}.mp4"
        media.write_video(video_path, np.stack(frames), fps=FPS, codec="h264")

        # store per-episode metrics ------------------------------------------
        metrics[TASK]["success"].append(int(success))
        metrics[TASK]["ep_len"].append(steps)
        metrics[TASK]["path_len"].append(path_len)
        metrics[TASK]["collisions"].append(collision_events)

        print(f"[{TASK}] ep {ep:02d} → "
              f"{'✓' if success else '✗'} | steps={steps:3d} | "
              f"path={path_len:6.3f} m | coll={collision_events:2d} | "
              f"saved→ {video_path}")

# ---------------------------------------------------------------------------
# 3)  Final aggregated report ------------------------------------------------
for TASK in TASKS:
    succ   = np.mean(metrics[TASK]["success"])
    elen   = np.mean(metrics[TASK]["ep_len"])
    plen   = np.mean(metrics[TASK]["path_len"])
    coll   = np.mean(metrics[TASK]["collisions"])

    print("\n" + "-"*60)
    print(f"{TASK}")
    print("-"*60)
    print(f" Success rate   : {succ:6.3f} ({sum(metrics[TASK]['success'])}"
          f"/{N_EPISODES})")
    print(f" Episode length : {elen:6.2f} steps (avg)")
    print(f" Path length    : {plen:6.3f} m   (avg)")
    print(f" Collisions     : {coll:6.2f} contacts (avg)")
print("-"*60)
