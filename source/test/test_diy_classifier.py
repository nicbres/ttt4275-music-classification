import numpy as np
import pytest

import source.data_handling
import source.mappings
import source.plotting
import source.diy_classifiers


@pytest.mark.parametrize("inputs, order, expected",
    [
        [np.array([1, 2]), 1, 3],
        [np.array([3, 4]), 2, 5],
    ],
)
def test_p_norm(inputs, order, expected):
    result = source.diy_classifiers.p_norm(p=order, vec=inputs)

    assert abs(result - expected) < 1e-9

