import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.models import model_forceldm as _model_forceldm


def make_yuanluo_example() -> dict:
    """Creates a random input example for the Reactive Diffusion Policy."""
    return {
        "observation/front_camera": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/left_wrist_camera": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/gelsight_left": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        # "observation/gelsight_right": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/left_wrench": np.random.rand(6),
        "observation/left_state": np.random.rand(8),
        "actions": np.random.rand(8),
        "prompt": "Please Insert the USB hub into the USB slot on the board.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YuanluoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for the Reactive Diffusion Policy dataset.
    It is used for both training and inference.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    state_dim: int = 7
    # Model image slot -> camera field in the dataset / inference observation.
    camera_mapping: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "base_0_rgb": "observation.images.head_camera",
            "left_wrist_0_rgb": "observation.images.wrist_left_camera",
        }
    )

    def __post_init__(self):
        if not self.camera_mapping or not set(self.camera_mapping).issubset(_model_forceldm.IMAGE_KEYS):
            raise ValueError(f"camera_mapping must use model image slots from {_model_forceldm.IMAGE_KEYS}.")

    def __call__(self, data: dict) -> dict:
        images = {}
        for image_key, camera_key in self.camera_mapping.items():
            if camera_key not in data:
                raise ValueError(f"Configured camera `{camera_key}` for `{image_key}` is missing.")
            images[image_key] = _parse_image(data[camera_key])

        inputs = {
            "state": data["observation.state"][:self.state_dim],
            "image": images,
            "image_mask": {key: np.True_ for key in images},
        }
        if "observation.images.future_flow" in data:
            inputs["flow_img"] = _parse_image(data["observation.images.future_flow"])

        # Pad actions to the model action dimension. Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            raise ValueError("No task prompt found!")

        return inputs


@dataclasses.dataclass(frozen=True)
class YuanluoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format for Reactive Diffusion Policy.
    It is used for inference only.
    """
    action_dim = 7 

    def __call__(self, data: dict) -> dict:
        # Return the actions with the correct dimension.
        # Assuming 7 actions (e.g., 3 for position, 4 for quaternion).
        # This needs to match the actual action dimension of your robot.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
    

@dataclasses.dataclass(frozen=True)
class ForceLDMInputs(YuanluoInputs):
    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        inputs["effort"] = data["observation.effort"] # Append 6-axis force sensor data 
        if "observation.images.future_flow" in data:
            inputs["flow_img"] = _parse_image(data["observation.images.future_flow"])
        if "observation.images.future_wrist_flow" in data:
            inputs["wrist_flow_img"] = _parse_image(data["observation.images.future_wrist_flow"])
        
        # Optionally pass through contact-related supervision signals if present
        if "observation.is_contact" in data:
            inputs["is_contact"] = data["observation.is_contact"]
        return inputs

@dataclasses.dataclass(frozen=True)
class ForceLDMOutputs(YuanluoOutputs):
    pass
