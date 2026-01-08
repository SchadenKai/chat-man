import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "../data"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "paultimothymooney/recipenlg",
    file_path,
)