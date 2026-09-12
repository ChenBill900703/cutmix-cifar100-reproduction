import random

import numpy as np
import torch

from cutmix_repro.runtime import capture_rng_state, make_numpy_rng, restore_rng_state


def test_rng_roundtrip_cpu():
    random.seed(3); np.random.seed(3); torch.manual_seed(3)
    generator = make_numpy_rng(3)
    state = capture_rng_state(generator)
    expected = (random.random(), np.random.random(), torch.rand(1), generator.random())
    restore_rng_state(state, generator)
    actual = (random.random(), np.random.random(), torch.rand(1), generator.random())
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
    assert expected[3] == actual[3]

