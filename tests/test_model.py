import copy
import io
from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from adcf_yolo.build import adcf_layers, apply_adcf, make_trainer, neck_slots
from adcf_yolo.modules import ADCF, FUSION_MODES, ResidualRefine

P2_CFG = "configs/models/yolo11n-p2.yaml"


@pytest.mark.parametrize("fusion", FUSION_MODES)
@pytest.mark.parametrize("gate", ["spatial", "channel"])
@pytest.mark.parametrize("highpass", [True, False])
def test_adcf_shapes_and_gradients(fusion, gate, highpass):
    m = ADCF(96, 64, fusion=fusion, gate=gate, highpass=highpass)
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
    model = DetectionModel("yolo11n.yaml", nc=8, verbose=False)
    with pytest.raises(ValueError):
        apply_adcf(copy.deepcopy(model), mode="nope")
    with pytest.raises(ValueError):
        apply_adcf(copy.deepcopy(model), placement="fine")  # P2-only slots
    with pytest.raises(ValueError):
        apply_adcf(copy.deepcopy(model), module="cbam", mode="replace")


@pytest.mark.parametrize("cfg", ["yolov8n.yaml", "yolo11n.yaml", "yolo26n.yaml", P2_CFG, "yolo26n-p2.yaml"])
@pytest.mark.parametrize("spec", [
    {"placement": "all", "mode": "replace"},
    {"placement": "all", "mode": "residual"},
    {"placement": "p3", "mode": "residual"},
    {"placement": "all", "mode": "residual", "module": "cbam"},
])
def test_patch_yolo_neck(cfg, spec):
    model = apply_adcf(DetectionModel(cfg, nc=8, verbose=False), **spec)
    n_slots = len(neck_slots(model))
    expected = 0 if spec.get("module") == "cbam" else (n_slots if spec["placement"] == "all" else 1)
    assert len(adcf_layers(model)) == expected
    assert model.yaml["adcf"]["mode"] == spec["mode"]
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 320, 320)) is not None


def test_p2_config_parses_scale():
    model = DetectionModel(P2_CFG, nc=8, verbose=False)
    assert model.yaml["scale"] == "n"
    assert list(neck_slots(model)) == ["p4_td", "p3_td", "p2_out", "p3_out", "p4_out", "p5_out"]
    assert len(model.stride) == 4 and int(model.stride.min()) == 4


def test_residual_refine_starts_near_identity():
    block = torch.nn.Conv2d(16, 16, 3, padding=1)
    wrapped = ResidualRefine(block, ADCF(16, 16), 16, gamma_init=0.0).eval()
    x = torch.randn(1, 16, 12, 12)
    assert torch.equal(wrapped(x), block(x))


def _trainer(adcf):
    t = make_trainer(adcf).__new__(make_trainer(adcf))
    t.data = {"nc": 8, "channels": 3, "names": {i: str(i) for i in range(8)}}
    t.args = SimpleNamespace(cls_remap=True)
    return t


@pytest.mark.parametrize("mode", ["replace", "residual"])
def test_trainer_loads_pretrained_then_resumes(mode):
    pretrained = DetectionModel("yolo11n.yaml", nc=8, verbose=False)
    model = _trainer({"placement": "all", "mode": mode}).get_model(cfg=pretrained.yaml, weights=pretrained, verbose=False)
    i = neck_slots(model)["p3_out"]
    assert torch.equal(model.model[0].conv.weight, pretrained.model[0].conv.weight)  # backbone loaded
    if mode == "residual":  # pretrained neck block kept inside the wrapper
        assert torch.equal(model.model[i].block.cv1.conv.weight, pretrained.model[i].cv1.conv.weight)

    # Simulate training, then resume: the spec comes from the checkpoint's yaml and all weights restore.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.123)
    resumed = _trainer(None).get_model(cfg=model.yaml, weights=model, verbose=False)
    for (k, a), b in zip(model.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(a, b), k


def test_checkpoint_pickle_roundtrip():
    patched = apply_adcf(DetectionModel("yolo11n.yaml", nc=8, verbose=False), mode="residual")
    buf = io.BytesIO()
    torch.save({"model": patched}, buf)
    buf.seek(0)
    restored = torch.load(buf, weights_only=False)["model"]
    assert len(adcf_layers(restored)) == 4
