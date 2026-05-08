## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Expected Data Layout

Place MIDI files in:

- Required: `data/raw_midi/unknown/`

## 3) Quick Start

#### Task 1 (Autoencoder, 5 samples)
```bash
python -m src.training.train_ae --epochs 10 --genre unknown
python -m src.generation.generate_music --model ae --num-samples 5 --seq-len 128
```

#### Task 2 (VAE, 8 samples + latent interpolation)
```bash
python -m src.training.train_vae --epochs 10
python -m src.generation.generate_music --model vae --num-samples 8 --seq-len 128
python -m src.generation.latent_interpolation --steps 8 --seq-len 128
```

#### Task 3 (Transformer, 10 long-sequence samples)
```bash
python -m src.training.train_transformer --epochs 10
python -m src.generation.generate_music --model transformer --num-samples 10 --seq-len 128 --genre unknown
```

#### Baselines (required comparison)
```bash
python -m src.generation.generate_music --model random --num-samples 5 --seq-len 128
python -m src.generation.generate_music --model markov --num-samples 5 --seq-len 128
```

#### Task 4 (RLHF, 10 fine-tuned samples)
```bash
python -m src.training.train_rlhf --steps 100 --seq-len 128
python -m src.generation.generate_music --model rlhf --num-samples 10 --seq-len 128 --genre unknown
```

#### Build Evaluation Tables
```bash
python -m src.evaluation.build_comparison_table
python -m src.evaluation.rlhf_before_after
```

## 4) Outputs

- MIDI files: `outputs/generated_midis/`
- Training plots: `outputs/plots/`
- Metrics JSON files and summary CSV: `outputs/plots/`
- Human survey files: `outputs/survey_results/human_survey_template.csv`, `outputs/survey_results/human_survey_results.csv`