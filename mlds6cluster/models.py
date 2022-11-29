from typing import Protocol
from abc import abstractmethod
from numpy import ndarray as Array
from sklearn.cluster import (
        KMeans, SpectralClustering
        )
from sklearn.mixture import GaussianMixture
from enum import Enum

class ModelEnum(Enum):
    KMEANS = "KMEANS"
    SPECTRAL = "SPECTRAL"
    AFFINITY = "AFFINITY"
    GMM = "GMM"

models = {
        ModelEnum.KMEANS: KMeans,
        ModelEnum.SPECTRAL: SpectralClustering,
        ModelEnum.GMM: GaussianMixture
        }
predict_models = (
        ModelEnum.KMEANS, ModelEnum.GMM
        )

class BaseEstimator(Protocol):
    labels_: Array

    @abstractmethod
    def fit(self, X: Array) -> "BaseEstimator":
        ...

    @abstractmethod
    def predict(self, X: Array) -> Array:
        ...

class ModelBuilder:
    model: BaseEstimator
    def __init__(
            self,
            model_type: ModelEnum,
            n_clusters: int,
            ):
        self.model_type = model_type
        self.n_clusters = n_clusters

    def build(self) -> "ModelBuilder":
        self.model = models[self.model_type](self.n_clusters)
        return self

    def train(self, X: Array) -> "ModelBuilder":
        self.model.fit(X)
        return self

    def predict(self, X: Array) -> Array: 
        if self.model_type in predict_models:
            y = self.model.predict(X)
        else:
            y = self.model.labels_
        return y
