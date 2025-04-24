from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv


class EcoTInference:
    def __init__(
        self,
        saved_model_path: str = "Embodied-CoT/ecot-openvla-7b-bridge",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 1,
        pred_action_horizon: int = 1,
        exec_horizon: int = 1,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # set default unnormalization key and sticky gripper repeats
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Unsupported policy setup: {policy_setup}")
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        # load processor and model
        self.processor = AutoProcessor.from_pretrained(saved_model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()

        # inference settings
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        # initial gripper and task state
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.task_description = None
        self.num_image_history = 0

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        # reset if new task
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8, "Expected uint8 image"
        image = self._resize_image(image)
        img_pil = Image.fromarray(image)

        # prepare inputs
        inputs = self.processor(task_description, img_pil).to("cuda:0", dtype=torch.bfloat16)
        # predict: EcoT returns (actions, reasoning_ids)
        result = self.model.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )
        # unpack tuple if chain-of-thought is returned
        if isinstance(result, tuple) or isinstance(result, list):
            raw_actions_array, reasoning_ids = result
        else:
            raw_actions_array = result
        # ensure numpy array
        raw_actions_array = np.array(raw_actions_array)
        # add batch dim
        batched = raw_actions_array[None]

        # build raw_action dict
        raw_action = {
            "world_vector": batched[0, :3],
            "rotation_delta": batched[0, 3:6],
            "open_gripper": batched[0, 6:7],
        }

        # process world motion
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        r, p, y = raw_action["rotation_delta"].astype(np.float64)
        ax, ang = euler2axangle(r, p, y)
        action["rot_axangle"] = (ax * ang) * self.action_scale

        # gripper logic
        if self.policy_setup == "google_robot":
            current = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                rel = np.array([0.0])
            else:
                rel = self.previous_gripper_action - current
            self.previous_gripper_action = current

            if np.abs(rel) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = rel
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                rel = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = rel
        else:
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(img) for img in images]
        labels = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(images[::3], axis=1)
        layout = [["image"] * len(labels), labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(layout)
        fig.set_size_inches([45, 10])

        pred = np.array([
            np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
            for a in predicted_raw_actions
        ])
        for i, lbl in enumerate(labels):
            axs[lbl].plot(pred[:, i], label="predicted")
            axs[lbl].set_title(lbl)
            axs[lbl].set_xlabel("Timestep")
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Timestep (subsampled)")
        plt.legend()
        plt.savefig(save_path)
