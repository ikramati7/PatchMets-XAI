import torch

from pcam_mets_explain.model import build_resnet18


def test_resnet18_binary_output_shape():
    model = build_resnet18(pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 96, 96))
    assert out.shape == (2, 1)
