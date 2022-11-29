from sklearn.datasets import (
        make_blobs, make_moons, make_circles
        )
from numpy import ndarray as Array
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, Dict

class ClusteringEnum(Enum):
    BLOBS = "BLOBS"
    CIRCLES = "CIRCLES"
    MOONS = "MOONS"

data_f: Dict[ClusteringEnum, Callable] = {
        ClusteringEnum.BLOBS: make_blobs,
        ClusteringEnum.CIRCLES: make_circles,
        ClusteringEnum.MOONS: make_moons
        }

class AbstractDataset(ABC):
    n_samples: int # type hinting
    seed: int
    noise: float
    raw_f: Callable
    f: Callable

    def __init__(
            self,
            dataset_type: ClusteringEnum,
            n_samples: int,
            noise: float,
            seed: int):
        self.n_samples = n_samples
        self.seed = seed
        self.noise = noise
        self.raw_f = data_f[dataset_type]

    @abstractmethod
    def init(self) -> None:
        ...

    def sample(self) -> Array:
        X, _ = self.f()
        return X

class BlobsDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(BlobsDataset, self).__init__(*args, **kwargs)

    def init(self):
        self.f = partial(
                self.raw_f,
                n_samples = self.n_samples,
                cluster_std = self.noise,
                random_state = self.seed
                )

class BiClusterDataset(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(BiClusterDataset, self).__init__(*args, **kwargs)

    def init(self):
        self.f = partial(
                self.raw_f,
                n_samples = self.n_samples,
                noise = self.noise,
                random_state = self.seed
                )

