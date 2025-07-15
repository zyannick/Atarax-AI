import pytest
import torch
from ataraxai.praxis.modules.prompt_engine.specific_tasks.image_captioning_task import (
    ImageCaptioningTask,
)


def test_init_default_values():
    task = ImageCaptioningTask()
    assert task.id == "image_captioning"
    assert task.description == "Generates a descriptive caption for a given image."
    assert task.required_inputs == ["image_path"]
    assert task.model_id == "llava-hf/llava-1.5-7b-hf"
    assert task.model is None
    assert task.processor is None
    assert hasattr(task, "device")
    assert task.device in ("cuda", "cpu")


def test_init_custom_model_id():
    custom_id = "custom/model"
    task = ImageCaptioningTask(model_id=custom_id)
    assert task.model_id == custom_id


def test_device_assignment(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    task = ImageCaptioningTask()
    assert task.device == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    task = ImageCaptioningTask()
    assert task.device == "cpu"
