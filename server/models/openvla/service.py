from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import io
import json
from PIL import Image
from transforms3d.euler import euler2axangle

# Import the OpenVLAInference class (ensure it's in your PYTHONPATH or same directory)
from openvla_inference import OpenVLAInference

app = FastAPI()
# Instantiate a single global inference engine
inference = OpenVLAInference(policy_setup='google_robot')

class UpdateParams(BaseModel):
    horizon: Optional[int]
    pred_action_horizon: Optional[int]
    exec_horizon: Optional[int]
    image_size: Optional[List[int]]
    action_scale: Optional[float]

@app.get("/ping")
def ping():
    """Health-check endpoint returning a simple status."""
    return {"status": "ok"}

@app.post("/update_inference_parameters")
def update_inference_parameters(params: UpdateParams):
    """
    Update inference parameters on the fly. Only provided fields will be updated.
    """
    updates = params.dict(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No parameters provided to update.")
    for key, value in updates.items():
        if not hasattr(inference, key):
            raise HTTPException(status_code=400, detail=f"Invalid parameter: {key}")
        setattr(inference, key, value)
    return JSONResponse(content={"status": "success", "updated": updates})

class ResetRequest(BaseModel):
    task_description: str

@app.post("/reset")
def reset(req: ResetRequest):
    """
    Reset the inference state with a new task description.
    """
    inference.reset(req.task_description)
    return {"status": "reset", "task_description": req.task_description}

@app.post("/step")
async def step(
    task_description: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    """
    Run one inference step on an uploaded image. Returns raw_action and processed action.
    """
    contents = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    print(task_description)

    raw_action, action = inference.step(image, task_description)
    # Convert numpy arrays to lists for JSON serialization
    def np_to_list(d):
        return {k: v.tolist() for k, v in d.items()}

    return {"raw_action": np_to_list(raw_action), "action": np_to_list(action)}

# @app.post("/step")
# async def step(
#     task_description: Optional[str] = Form(None),
#     file: UploadFile = File(...)
# ):
#     """
#     Run one inference step on an uploaded image. Returns raw_action and processed action.
#     """
#     contents = await file.read()
#     try:
#         image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image file.")
#     print(task_description)

#     return {"raw_action": [1, 2, 3, 4], "action": [4, 5, 6]}

@app.post("/visualize_epoch")
async def visualize_epoch(
    save_path: str = Form(...),
    files: List[UploadFile] = File(...),
    predicted: str = Form(...)
):
    """
    Generate and return a visualization of an epoch given predicted actions and image sequence.
    - save_path: where to save the output image (e.g. "/tmp/epoch.png")
    - files: sequence of image files
    - predicted: JSON string of list of dicts with keys ['world_vector','rotation_delta','open_gripper']
    """
    # Load images
    images = []
    for f in files:
        contents = await f.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="One or more files are not valid images.")
        images.append(np.array(img))

    # Parse predicted actions JSON
    try:
        pred_list = json.loads(predicted)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="`predicted` must be valid JSON.")

    # Convert to model's expected input
    predicted_raw_actions = []
    for item in pred_list:
        # Concatenate to a single array [7-dim]
        vec = np.concatenate([
            np.array(item["world_vector"]),
            np.array(item["rotation_delta"]),
            np.array(item["open_gripper"])], axis=-1)
        predicted_raw_actions.append(vec)

    # Call visualization
    inference.visualize_epoch(predicted_raw_actions, images, save_path)

    # Return the saved image
    return FileResponse(save_path, media_type="image/png")
