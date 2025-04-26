import simpler_env
from collections import defaultdict
import requests
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import numpy as np
import mediapy as media
import cv2
from pathlib import Path


class Experiment:
    def __init__(self, tasks: list[str], n_episodes, fps, prompts: list = [], experiment_name='experiment'):
        print(f"INITIALIZING {experiment_name}")
        self.tasks = tasks
        self.prompts = []
        for i in range(len(tasks)):
            if len(prompts) <= i or prompts[i] is None:
                env = simpler_env.make(tasks[i])
                self.prompts.append(env.get_language_instruction())
            else:
                self.prompts.append(prompts[i])
        self.n_episodes = n_episodes
        self.fps = fps
        self.experiment_name = experiment_name
        self.metrics = {t: defaultdict(list) for t in tasks}

    def run(self):
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            prompt = self.prompts[i]
            print(f"RUNNING {task} with prompt {prompt}")
            Path(f"{self.experiment_name}/{task}").mkdir(parents=True, exist_ok=True)
            env = simpler_env.make(task)

            for ep in range(self.n_episodes):
                print(f"TRYING {ep}/{self.n_episodes}")
                obs, _ = env.reset()
                scene = env.unwrapped._scene if hasattr(env.unwrapped, "_scene") else env.unwrapped.scene
                rlinks = self._hand_links(scene)
                prev_contact_count = 0
                frames = []
                prev_tcp = self._tcp_position(obs, env)
                steps, path_len, collision_events = 0, 0.0, 0

                # reset model
                response = requests.post(
                    "http://localhost:8003/reset",
                    data={"task_description": prompt}
                )

                success = trunc = False
                while not (success or trunc):
                    img = get_image_from_maniskill2_obs_dict(env, obs)
                    frames.append(img)

                    success_cv2, img_encoded_jpeg = cv2.imencode('.jpg', img)
                    if not success_cv2:
                        raise RuntimeError("Failed to encode image")

                    response = requests.post(
                        "http://localhost:8003/step",
                        files={
                            # The key "image" should match the parameter name in your API
                            "file": ("image.jpg", img_encoded_jpeg.tobytes(), "image/jpeg"),
                        },
                        data={
                            "task_description": prompt,
                        }
                    )

                    result = response.json()
                    act = result["action"]
                    vec = np.concatenate([act["world_vector"],
                                          act["rot_axangle"],
                                          act["gripper"]])
                    obs, _, success, trunc, info = env.step(vec)

                    # -------- metric accumulation -------
                    steps += 1

                    # Calculate path length

                    tcp = self._tcp_position(obs, env)
                    path_len += np.linalg.norm(tcp - prev_tcp)
                    prev_tcp = tcp

                    # cache once
                    if "scene" not in locals():
                        scene = env.unwrapped._scene if hasattr(env.unwrapped, "_scene") else env.unwrapped.scene
                        rlinks = self._hand_links(scene)

                    # compute current number of environment–robot contacts
                    curr_contact_count = self._count_robot_env_contacts(scene, rlinks)
                    # only count an “event” if it actually changed since last frame
                    if curr_contact_count > prev_contact_count:
                        collision_events += 1
                    prev_contact_count = curr_contact_count

                # save video ----------------------------------------------------------
                tag = "success" if success else "fail"
                video_path = f"{self.experiment_name}/{task}/episode_{ep:02d}_{tag}.mp4"
                media.write_video(video_path, np.stack(frames), fps=self.fps, codec="h264")

                # store per-episode metrics ------------------------------------------
                self.metrics[task]["success"].append(int(success))
                self.metrics[task]["ep_len"].append(steps)
                self.metrics[task]["path_len"].append(path_len)
                self.metrics[task]["collisions"].append(collision_events)

                print(f"[{task}] ep {ep:02d} → "
                      f"{'✓' if success else '✗'} | steps={steps:3d} | "
                      f"path={path_len:6.3f} m | coll={collision_events:2d} | "
                      f"saved→ {video_path}")

        for task in self.tasks:
            succ = np.mean(self.metrics[task]["success"])
            elen = np.mean(self.metrics[task]["ep_len"])
            plen = np.mean(self.metrics[task]["path_len"])
            coll = np.mean(self.metrics[task]["collisions"])

            print("\n" + "-" * 60)
            print(f"{task}")
            print("-" * 60)
            print(f" Success rate   : {succ:6.3f} ({sum(self.metrics[task]['success'])}"
                  f"/{self.n_episodes})")
            print(f" Episode length : {elen:6.2f} steps (avg)")
            print(f" Path length    : {plen:6.3f} m   (avg)")
            print(f" Collisions     : {coll:6.2f} contacts (avg)")
        print("-" * 60)

    def _hand_links(self, scene):
        """Subset of robot links that a task can physically collide with."""
        rlinks = self._robot_links(scene)
        return {lk for lk in rlinks
                if any(k in lk.get_name().lower() for k in ("gripper", "finger", "tcp", "hand", "wrist"))}

    def _tcp_position(self, obs, env):
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
            pass  # simply try the next method

        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "get_tcp_pose"):
            return np.asarray(env.unwrapped.get_tcp_pose()[:3])

        # 3) ---- fall back to SAPIEN gripper_tcp link -------------------------
        scene = getattr(env.unwrapped, "_scene", None) or getattr(env.unwrapped, "scene", None)
        if scene is not None:
            for art in scene.get_all_articulations():
                for link in art.get_links():
                    if "gripper_tcp" in link.get_name():
                        return np.asarray(link.get_pose().p)  # Pose → position

        raise RuntimeError("Could not locate TCP position in obs or env.")

    def _count_robot_env_contacts(self, scene, links):
        """Contacts where exactly one body is a robot link."""
        # print(links)
        n = 0
        for cp in scene.get_contacts():  # list[Contact]
            a0, a1 = cp.actor0, cp.actor1
            in0, in1 = a0 in links, a1 in links
            if in0 ^ in1:  # XOR → env-robot, not self-collision
                n += 1
                # print(a0, a1)
        return n

    def _robot_links(self, scene):
        """Return the set of links that belong to the *first* SAPIEN articulation."""
        arts = scene.get_all_articulations()
        return set(arts[0].get_links()) if arts else set()
