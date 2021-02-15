from dataclasses import dataclass
from typing import Optional

import numpy as np

from dqn.datum import NpDef, SerializableNpData


@dataclass
class A(SerializableNpData):
    a: np.ndarray
    b: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.a)


a_def = {'a': NpDef(shape=(2, ), dtype=np.float32), 'b': None}


@dataclass
class B(SerializableNpData):
    a: np.ndarray
    b: np.ndarray

    def __len__(self) -> int:
        return len(self.a)


b_def = {'a': NpDef(shape=(2, ), dtype=np.float32), 'b': NpDef(shape=(1, 2, 3), dtype=np.bool)}


def test_serializable_np_data():
    A.as_random(a_def, size=2).validate_type(a_def)
    a = A.as_random(a_def, size=2)
    assert a.eq(other=A.from_bytes(bytes_data=a.to_bytes(np_defs=a_def), np_defs=a_def), np_defs=a_def)

    B.as_random(b_def, size=2).validate_type(b_def)
    b = A.as_random(b_def, size=2)
    assert b.eq(other=B.from_bytes(bytes_data=b.to_bytes(np_defs=b_def), np_defs=b_def), np_defs=b_def)
