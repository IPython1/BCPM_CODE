# BCPM

#### Disentangling Auxiliary Behaviors for MOOC Video Recommendation

This repository provides the official PyTorch implementation of **BCPM** for MOOC video recommendation on multi-behavior data (browse, favorite, study). The model disentangles auxiliary behaviors to improve target-behavior (study) prediction.

---

#### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#### Getting Started

**1. Prepare the dataset**

Place preprocessed MOOC data under `./data/`. Each dataset folder should contain at least:

| File | Description |
|------|-------------|
| `count.txt` | User / item counts (`user`, `item`) |
| `pos_sampling.txt` | Training positive samples |
| `browse.txt`, `favorite.txt`, `study.txt` | Interaction list per behavior |
| `browse_dict.txt`, `favorite_dict.txt`, `study_dict.txt` | User → item list per behavior (JSON) |
| `all_dict.txt` | All training interactions per user |
| `validation_dict.txt` | Validation ground truth |
| `test_dict.txt` | Test ground truth |

Datasets are not bundled in this repository due to their size. Download the raw data (see [Dataset Download](#dataset-download)), extract under `./data/`, then run `data_process.py` in the corresponding subdirectory to generate the files above.

**2. Create checkpoint directory**

```bash
mkdir check_point
```

Model checkpoints are saved to `./check_point/` by default.

**3. Train and evaluate**

MOOCCube:

```bash
python main.py --data_name mooccube --device cuda:0
```

MOOCCubeX:

```bash
python main.py --data_name mooccubex --device cuda:0
```

Dual-GPU training:

```bash
python main.py --data_name mooccube --gpu_no 2 --device cuda:0
```

Training logs are written to `./log/BCPM/`.

---

#### Supported Datasets

| `--data_name` | Data path | Behaviors |
|---------------|-----------|-----------|
| `mooccube` | `./data/MOOCCube` | browse, favorite, study |
| `mooccubex` | `./data/MOOCCubeX` | browse, favorite, study |

Auxiliary behaviors: **browse**, **favorite**. Target behavior: **study**.

---

#### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding_size` | 64 | Embedding dimension |
| `--lr` | 0.001 | Learning rate |
| `--reg_weight` | 1e-3 | Embedding regularization |
| `--log_reg` | 0.5 | Weight of BCE vs BPR loss |
| `--batch_size` | 1024 | Training batch size |
| `--epochs` | 200 | Max epochs (early stopping enabled) |
| `--neg_sample_size` | 99 | Negatives per positive in sampled evaluation |
| `--device` | cuda:0 | Device for training |
| `--gpu_no` | 1 | Use `2` for DataParallel on two GPUs |
| `--model_name` | BCPM | Log and checkpoint name prefix |
| `--topk` | [5, 10] | Top-K for hit / ndcg metrics |

---

#### Evaluation Protocol

Validation and test use **sampled negative evaluation**: for each ground-truth pair, the model ranks **1 positive + `neg_sample_size` random negatives** (default 100 candidates). Metrics reported include `hit@K` and `ndcg@K` (`K` ∈ {5, 10}).

---

#### Project Structure

```
BCPM_CODE/
├── main.py           # Entry point and hyperparameters
├── model.py          # BCPM model
├── data_set.py       # Dataset and sampled test loader
├── trainer.py        # Single-GPU trainer
├── trainer_2GPU.py   # Dual-GPU trainer
├── lightGCN.py       # LightGCN propagation
├── metrics.py        # Hit, NDCG, etc.
├── utils.py          # Loss functions
├── requirements.txt
└── data/             # MOOCCube / MOOCCubeX (download separately)
```

---

#### Dataset Download

Datasets are too large to ship with this repo. Download and place them under `./data/`:

| Dataset | Download |
|---------|----------|
| MOOCCube (raw) | http://moocdata.cn/data/MOOCCube |
| MOOCCubeX | https://github.com/THU-KEG/MOOCCubeX |

For `mooccube`, place the raw MOOCCube files under `./data/MOOCCube/`, run `python data/data_process.py`, then train with `--data_name mooccube`. For `mooccubex`, place files under `./data/MOOCCubeX/` and run `python data/data_process.py MOOCCubeX`.

---

#### Citation

If you use this code, please cite:

```bibtex
@article{bcpm,
  title={BCPM: Disentangling Auxiliary Behaviors for MOOC Video Recommendation},
  author={},
  year={}
}
```
