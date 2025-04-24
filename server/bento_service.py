import numpy as np
from PIL import Image
import bentoml
from bentoml.io import Image as ImageIO, JSON
from openvla_inference import OpenVLAInference

# 1) Create a Runner from your class
openvla_runner = bentoml.Runner(
    OpenVLAInference,
    name="openvla_runner",
    init_parameters={
        "saved_model_path": "openvla/openvla-7b",
        "policy_setup": "widowx_bridge",      # or "google_robot"
        # (other init args if you wish)
    },
)

# 2) Define a Service that uses that Runner
service = bentoml.Service("openvla_service", runners=[openvla_runner])

@service.api(
    input=ImageIO(),       # accepts image/* upload
    output=JSON()          # returns JSON
)
async def predict(image: Image.Image):
    """
    Expects:
      - an image file
      - a 'task_description' query parameter in the URL, e.g.
          POST /predict?task_description=pick+up+the+red+cube
    """
    # parse the task_description from query params
    import starlette.requests
    req: starlette.requests.Request = bentoml.context.get_request()
    task_desc = req.query_params.get("task_description", "")
    # convert to numpy array
    img_np = np.array(image.convert("RGB"), dtype=np.uint8)
    # run the model
    raw_action, action = await openvla_runner.run(img_np, task_desc)
    # serialize to lists for JSON
    return {
        "raw_action": {k: v.tolist() for k, v in raw_action.items()},
        "action":     {k: v.tolist() for k, v in action.items()},
    }
