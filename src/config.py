from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.tokenizer import VOCAB_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "train_test_split"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MIDI_OUTPUT_DIR = OUTPUT_DIR / "generated_midis"
PLOT_DIR = OUTPUT_DIR / "plots"
SURVEY_DIR = OUTPUT_DIR / "survey_results"


GENRES = ["unknown"]
GENRE_TO_ID = {name: idx for idx, name in enumerate(GENRES)}


@dataclass
class TrainConfig:
    seq_len: int = 128 #sequence length
    vocab_size: int = VOCAB_SIZE #vocabulary size
    latent_dim: int = 64 #latent dimension
    hidden_dim: int = 256 #hidden dimension
    embed_dim: int = 128 #embedding dimension
    num_layers: int = 2 #number of layers
    batch_size: int = 32 #batch size
    lr: float = 1e-3 #learning rate
    epochs: int = 10 #epochs
    beta: float = 0.01 #beta    
    train_split: float = 0.8 #train split
    seed: int = 42 #seed
    num_genres: int = len(GENRES) #number of genres


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        RAW_MIDI_DIR,
        PROCESSED_DIR,
        SPLIT_DIR,
        OUTPUT_DIR,
        MIDI_OUTPUT_DIR,
        PLOT_DIR,
        SURVEY_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
