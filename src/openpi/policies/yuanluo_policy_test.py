import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model_forceldm
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_forceldm import Pi0ForceLDM
from openpi.policies import yuanluo_policy
from openpi.training import config


def camera_mapping(num_views):
    mapping = {
        "base_0_rgb": "observation.images.head_camera",
        "left_wrist_0_rgb": "observation.images.wrist_left_camera",
    }
    if num_views == 3:
        mapping["right_wrist_0_rgb"] = "observation.images.side_camera"
    return mapping


def raw_sample(num_views):
    sample = {
        key: np.full((8, 8, 3), 40 * (index + 1), dtype=np.uint8)
        for index, key in enumerate(camera_mapping(num_views).values())
    }
    sample.update({
        "observation.state": np.zeros(7, dtype=np.float32),
        "observation.effort": np.zeros((42, 6), dtype=np.float32),
        "observation.is_contact": np.zeros(1, dtype=np.float32),
        "observation.future_flow.base_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation.future_flow.left_wrist_0_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "action": np.zeros((32, 7), dtype=np.float32),
        "task": "Insert the plug.",
    })
    return sample


@pytest.mark.parametrize("num_views", [2, 3])
def test_camera_mapping_training_and_inference(tmp_path, monkeypatch, num_views):
    # This test isolates camera routing; tokenization is covered by the real-data check.
    monkeypatch.setattr(config, "ModelTransformFactory", lambda **kwargs: lambda model: transforms.Group())
    cfg = config.get_config("pi0_forceldm")
    factory = dataclasses.replace(cfg.data, camera_mapping=camera_mapping(num_views))
    data_cfg = factory.create(tmp_path, cfg.model)
    training = transforms.compose([*data_cfg.repack_transforms.inputs, *data_cfg.data_transforms.inputs])(raw_sample(num_views))
    raw = raw_sample(num_views)
    raw["prompt"] = raw.pop("task")
    raw.pop("action")
    inference = yuanluo_policy.ForceLDMInputs(
        model_type=cfg.model.model_type, camera_mapping=camera_mapping(num_views)
    )(raw)
    assert set(training["image"]) == set(inference["image"]) == set(camera_mapping(num_views))
    assert set(training["image_mask"]) == set(training["image"])
    for key in training["image"]:
        np.testing.assert_array_equal(training["image"][key], inference["image"][key])
        assert training["image_mask"][key]


@pytest.mark.parametrize("num_views", [2, 3])
def test_all_cameras_reach_visual_prefix_and_attention(num_views):
    cfg = config.get_config("pi0_forceldm")
    raw = raw_sample(num_views)
    raw["prompt"] = raw.pop("task")
    inputs = yuanluo_policy.ForceLDMInputs(
        model_type=cfg.model.model_type, camera_mapping=camera_mapping(num_views)
    )(raw)
    obs = model_forceldm.Observation.from_dict({
        "image": {key: value[None] for key, value in inputs["image"].items()},
        "image_mask": {key: np.asarray([value]) for key, value in inputs["image_mask"].items()},
        "state": inputs["state"][None],
        "effort": inputs["effort"][None],
    })
    obs = model_forceldm.preprocess_observation(None, obs, effort_type=cfg.model.effort_type)
    assert set(obs.images) == set(camera_mapping(num_views))
    calls = []

    def encode_image(image, *, train):
        calls.append(image)
        values = jnp.mean(image, axis=(1, 2, 3))
        return jnp.broadcast_to(values[:, None, None], (1, 2, 4)), None

    model = SimpleNamespace(PaliGemma=SimpleNamespace(img=encode_image))
    tokens, mask, ar_mask = Pi0ForceLDM.embed_prefix(model, obs)
    assert len(calls) == num_views
    assert tokens.shape == (1, num_views * 2, 4)
    assert mask.shape == (1, num_views * 2)
    attention = make_attn_mask(mask, ar_mask)
    assert attention.shape == (1, num_views * 2, num_views * 2)
    assert np.asarray(attention).all()
    if num_views == 3:
        assert not np.allclose(np.asarray(tokens[:, :2]), np.asarray(tokens[:, -2:]))


def test_configured_camera_must_be_present():
    cfg = config.get_config("pi0_forceldm")
    transform = yuanluo_policy.ForceLDMInputs(model_type=cfg.model.model_type, camera_mapping=camera_mapping(3))
    with pytest.raises(ValueError, match="observation.images.side_camera"):
        transform(raw_sample(2))


@pytest.mark.parametrize("mapping", [{}, {"unknown_slot": "observation.images.side_camera"}])
def test_invalid_camera_mapping(mapping):
    cfg = config.get_config("pi0_forceldm")
    with pytest.raises(ValueError, match="camera_mapping"):
        yuanluo_policy.ForceLDMInputs(model_type=cfg.model.model_type, camera_mapping=mapping)
