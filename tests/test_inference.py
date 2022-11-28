import sys
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

sys.path.append('../')
from tabular_hypernet.hypernetwork import Hypernetwork

def test_inference():
    np.random.seed(42)
    torch.manual_seed(42)
    expected = torch.tensor([[-0.0534],
          [-0.0704],
          [-0.0618],
          [-0.0389],
          [-0.0526],
          [-0.0580],
          [-0.0544],
          [-0.0563],
          [-0.0682],
          [-0.0524]])

    data = torch.randn(10, 10)
    net = Hypernetwork(
        10,
        1,
        5,
        device="cpu"
    )
    res = net(data, net.test_mask[:1])
    assert res.shape == (10, 1)
    logger.info(res)
    assert torch.allclose(res, expected, rtol=1e-4, atol=1e-4), res

def test_test_inference():
    np.random.seed(42)
    torch.manual_seed(42)
    expected = torch.tensor([[-0.0103],
          [-0.0026],
          [-0.0442],
          [-0.0201],
          [-0.0155],
          [-0.0129],
          [-0.0428],
          [-0.0322],
          [-0.0150],
          [-0.0188]])

    data = torch.randn(10, 10)
    net = Hypernetwork(
        10,
        1,
        5,
        device="cpu"
    )
    net.eval()
    res = net(data)
    assert res.shape == (10, 1)
    logger.info(res)
    assert torch.allclose(res, expected, rtol=1e-4, atol=1e-4), res