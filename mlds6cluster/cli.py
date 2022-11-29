from argparse import ArgumentParser
from mlds6cluster.data import ClusteringEnum, DatasetBuilder
from mlds6cluster.models import ModelEnum, ModelBuilder
from mlds6cluster.viz import VizShower

def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="CLI para obtener imágenes de clustering con distintos datasets.")
    parser.add_argument(
            "--ds", type=ClusteringEnum, required=True,
            help="Dataset a usar.",
            choices=list(ClusteringEnum)
            )
    parser.add_argument(
            "--model", type=ModelEnum, required=True,
            help="Modelo a usar.",
            choices=list(ModelEnum)
            )
    parser.add_argument(
            "--k", type=int, default=2,
            help="Número de clusters."
            )
    parser.add_argument(
            "--n_samples", type=int, default=1000,
            help="Número de muestras."
            )
    parser.add_argument(
            "--noise", type=float, default=0.1,
            help="Nivel de ruido."
            )
    parser.add_argument(
            "--seed", type=int, default=42,
            help="Semilla aleatoria"
            )
    parser.add_argument(
            "--seconds", type=float, default=5.,
            help="Número de segundos a mostrar la imagen."
            )
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    X = (
            DatasetBuilder(
                dataset_kind = args.ds,
                n_samples = args.n_samples,
                noise = args.noise,
                seed = args.seed
                )
            .build()
            .sample()
            )
    y = (
            ModelBuilder(
                model_type = args.model,
                n_clusters = args.k
                )
            .build()
            .train(X)
            .predict(X)
            )
    VizShower(args.seconds).add_data(X, y).show()
