import io

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from adcf_yolo.build import PLACEMENTS, adcf_layers, apply_adcf, neck_fusion_indices
from adcf_yolo.modules import ADCF, FUSION_MODES


@pytest.mark.parametrize("fusion", FUSION_MODES)
@pytest.mark.parametrize("gate", ["spatial", "channel"])
def test_adcf_shapes_and_gradients(fusion, gate):
    m = ADCF(96, 64, fusion=fusion, gate=gate)
    x = torch.randn(2, 96, 20, 24, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 64, 20, 24)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gate_starts_balanced():
    m = ADCF(32, 32, fusion="gated").eval()
    m.store_alpha = True
    m(torch.randn(1, 32, 8, 8))
    assert m.last_alpha.shape == (1, 1, 8, 8)
    assert torch.allclose(m.last_alpha, torch.full_like(m.last_alpha, 0.5))


def test_invalid_options():
    with pytest.raises(ValueError):
        ADCF(8, 8, fusion="nope")
    with pytest.raises(ValueError):
        ADCF(8, 8, gate="nope")


@pytest.mark.parametrize("cfg", ["yolov8n.yaml", "yolo11n.yaml", "yolo26n.yaml"])
@pytest.mark.parametrize("placement", list(PLACEMENTS))
def test_patch_yolo_neck(cfg, placement):
    model = DetectionModel(cfg, nc=8, verbose=False)
    model = apply_adcf(model, placement=placement)
    assert len(adcf_layers(model)) == len(PLACEMENTS[placement])
    assert model.yaml["adcf"]["placement"] == placement
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 320, 320))
    assert out is not None


def test_checkpoint_roundtrip_and_pretrained_load():
    base = DetectionModel("yolo11n.yaml", nc=8, verbose=False)
    patched = apply_adcf(DetectionModel("yolo11n.yaml", nc=8, verbose=False))
    n_before = patched.model[0].conv.weight.clone()
    patched.load(base, verbose=False)  # baseline weights load into non-ADCF layers only
    assert torch.equal(patched.model[0].conv.weight, base.model[0].conv.weight)
    assert not torch.equal(n_before, base.model[0].conv.weight)

    buf = io.BytesIO()
    torch.save({"model": patched}, buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=False)["model"]
    assert isinstance(restored.model[neck_fusion_indices(restored)[0]], ADCF)
