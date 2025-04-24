import bentoml
from bentoml import Context
from bentoml.io import JSON, NumpyNdarray
import numpy as np

from openvla_inference import OpenVLAInference

# ---- lazy model init (fork-friendly) --------------------------------------
@bentoml.service(
    traffic={"timeout": 600},   # large models take time on cold start
)
class OpenVLAService:
    def __init__(self):
        self.model = OpenVLAInference()

    # ---------- API 1: update model hyper-params at runtime -----------------
    @bentoml.api
    def update_inference_parameters(self, params: dict) -> dict:
        """
        Example body: {"horizon": 3, "action_scale": 0.8}
        """
        for k, v in params.items():
            if hasattr(self.model, k):
                setattr(self.model, k, v)
        return {"status": "ok", "new_params": params}

    # ---------- API 2: reset episode / task --------------------------------
    @bentoml.api
    def reset(self, body: dict) -> dict:
        self.model.reset(body.get("task_description", ""))
        return {"status": "reset"}

    # ---------- API 3: step -------------------------------------------------
    @bentoml.api
    def step(self, image: np.ndarray, ctx: Context) -> dict:
        """
        Note: renamed `task` → `ctx` and imported `Context` to access headers properly.
        """
        # Use ctx.request.headers to pull through your custom header
        raw, action = self.model.step(
            image, ctx.request.headers.get("Task-Description")
        )
        return {"raw_action": raw, "action": action}

    # ---------- API 4: visualize one epoch ---------------------------------
    @bentoml.api
    def visualize_epoch(self, body: dict) -> dict:
        save_path = "/tmp/epoch.png"
        self.model.visualize_epoch(
            body["predicted_raw_actions"], body["images"], save_path
        )
        return {"plot": save_path}
