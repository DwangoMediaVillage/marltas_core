"""Base class of NumPy datum which can be shared through bytes data."""
from dataclasses import dataclass
from typing import Tuple, Type, TypeVar

import numpy as np

T = TypeVar('T', bound='Parent')


@dataclass
class NpDef:
    """Definition of `SerializableNpData`.

    Args:
        shape: Shape of NumPy array.
        dype: Data type of array.
    """
    shape: Tuple
    dtype: np.dtype
    byte_size: int = 0

    def __post_init__(self):
        self.byte_size = int(np.dtype(self.dtype).itemsize * np.prod(self.shape))


class SerializableNpData:
    """Base class"""
    def mem_size(self, np_defs: dict) -> int:
        size = self.__len__()
        return sum([np_def.byte_size * size for np_def in np_defs.values() if np_def is not None])

    @property
    def size(self) -> int:
        return self.__len__()

    def __len__(self) -> int:
        raise NotImplementedError

    def to_bytes(self, np_defs: dict) -> bytes:
        bytes_data = np.array(self.__len__(), dtype=np.uint32).tobytes()
        for name, np_def in np_defs.items():
            if np_def is not None:
                bytes_data += getattr(self, name).tobytes()
        return bytes_data

    @classmethod
    def from_bytes(cls: Type[T], np_defs: dict, bytes_data: bytes) -> T:
        size = int(np.frombuffer(bytes_data[:4], dtype=np.uint32))
        head = 4
        data = {}
        for name, np_def in np_defs.items():
            if np_def is not None:
                read_size = np_def.byte_size * size
                data[name] = np.frombuffer(bytes_data[head:head + read_size],
                                           dtype=np_def.dtype).reshape([size] + list(np_def.shape))
                head += read_size
        return cls(**data)

    @classmethod
    def as_random(cls: Type[T], np_defs: dict, size: int) -> T:
        data = {}
        for name, np_def in np_defs.items():
            if np_def is not None:
                shape = [size] + list(np_def.shape)
                data[name] = np.random.random(shape).astype(np_def.dtype)
        return cls(**data)

    @classmethod
    def as_empty(cls: Type[T], np_defs: dict, size: int) -> T:
        data = {}
        for name, np_def in np_defs.items():
            if np_def is not None:
                shape = [size] + list(np_def.shape)
                data[name] = np.empty(shape, dtype=np_def.dtype)
        return cls(**data)

    @classmethod
    def as_zeros(cls: Type[T], np_defs: dict, size: int) -> T:
        data = {}
        for name, np_def in np_defs.items():
            if np_def is not None:
                shape = [size] + list(np_def.shape)
                data[name] = np.zeros(shape=shape, dtype=np_def.dtype)
        return cls(**data)

    def validate_type(self, np_defs: dict) -> None:
        size = self.__len__()
        for name, np_def in np_defs.items():
            if np_def is not None:
                shape = tuple([size] + list(np_def.shape))
                v = getattr(self, name)
                assert v.shape == shape, f"{v.shape} != {shape}"
                assert v.dtype == np_def.dtype, f"{v.dtype} != {np_def.dtype}"

    def eq(self, other: T, np_defs: dict) -> bool:
        return all([
            np.array_equal(getattr(self, name), getattr(other, name)) if np_def is not None else True
            for name, np_def in np_defs.items()
        ])
