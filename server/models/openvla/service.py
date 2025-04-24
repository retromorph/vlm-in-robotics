import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np

from server.services.openvla.openvla_inference import OpenVLAInference

# ---- lazy model init (fork-friendly) --------------------------------------
@bentoml.service(
    traffic={"timeout": 600},   # large models take time on cold start
)
class OpenVLAService:
    def __init__(self):
        self.model = OpenVLAInference()

    # ---------- API 1: update model hyper-params at runtime -----------------
    @bentoml.api(input=JSON(), output=JSON())
    def update_inference_parameters(self, params: dict) -> dict:
        """
        Example body: {"horizon": 3, "action_scale": 0.8}
        """
        for k, v in params.items():
            if hasattr(self.model, k):
                setattr(self.model, k, v)
        return {"status": "ok", "new_params": params}

    # ---------- API 2: reset episode / task --------------------------------
    @bentoml.api(input=JSON(), output=JSON())
    def reset(self, body: dict) -> dict:
        self.model.reset(body.get("task_description", ""))
        return {"status": "reset"}

    # ---------- API 3: step -------------------------------------------------
    @bentoml.api(input=NumpyNdarray(dtype="uint8", shape=(-1,-1,3)), output=JSON())
    def step(self, image: np.ndarray, task: bentoml.Context) -> dict:   # context gives headers/query
        raw, action = self.model.step(image, task.headers.get("Task-Description"))
        return {"raw_action": raw, "action": action}

    # ---------- API 4: visualize one epoch ---------------------------------
    @bentoml.api(input=JSON(), output=JSON())
    def visualize_epoch(self, body: dict) -> dict:
        save_path = "/tmp/epoch.png"
        self.model.visualize_epoch(body["predicted_raw_actions"], body["images"], save_path)
        return {"plot": save_path}
