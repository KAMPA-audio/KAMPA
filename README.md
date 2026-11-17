# KAMPA: Knowledge Amalgamation with Meta-classifier-guided Pseudo-label Assignment

Code, data lists and released weights for KAMPA, a nine-class classifier for residential and
environmental sounds that reuses eight fixed binary teachers when only a few new labels are available.

Su Jin Ahn, Haeng Woon Kim, Ji Su Park (manuscript in preparation)

## Method in short

1. Eight binary teachers (EfficientAT mn10, one per named sound) score every clip.
2. A meta-classifier fitted on the same labeled clips as the student turns the eight teacher scores into a
   nine-class distribution (`model/make_combiner_labels.py`).
3. Each selected unlabeled clip receives one pseudo-label or two candidate labels, weighted by confidence.
4. A single student (EfficientAT mn10 backbone with a linear head) is trained on labeled clips plus these
   pseudo-labels and is the only model used at inference (`model/train_mark5.py`).

Classes: `heavy_impact`, `dragging`, `construction`, `machine_noise`, `speech`, `water_toilet`,
`water_shower`, `dog_bark`, `others` (see [`data/README.md`](data/README.md)).

## Results

Test set (430 clips), mean ± standard deviation over seeds 1-5, argmax over the nine class
probabilities for every model. Comparison models are trained on the labeled clips only.

| Labeled clips | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| 200 | KAMPA | 0.9316 ± 0.0102 | 0.9341 ± 0.0096 | 0.9316 ± 0.0102 | 0.9314 ± 0.0098 |
| 200 | EfficientAT | 0.8628 ± 0.0188 | 0.8657 ± 0.0189 | 0.8628 ± 0.0188 | 0.8601 ± 0.0202 |
| 200 | CED-Tiny | 0.8577 ± 0.0125 | 0.8601 ± 0.0117 | 0.8577 ± 0.0124 | 0.8561 ± 0.0120 |
| 200 | PANNs | 0.8014 ± 0.0172 | 0.8047 ± 0.0113 | 0.8014 ± 0.0172 | 0.7932 ± 0.0210 |
| 2000 | KAMPA | 0.9679 ± 0.0010 | 0.9691 ± 0.0014 | 0.9680 ± 0.0010 | 0.9682 ± 0.0012 |
| 2000 | EfficientAT | 0.9642 ± 0.0039 | 0.9651 ± 0.0038 | 0.9643 ± 0.0039 | 0.9643 ± 0.0039 |
| 2000 | CED-Tiny | 0.9544 ± 0.0063 | 0.9558 ± 0.0057 | 0.9545 ± 0.0063 | 0.9545 ± 0.0065 |
| 2000 | PANNs | 0.9056 ± 0.0194 | 0.9098 ± 0.0174 | 0.9057 ± 0.0194 | 0.9040 ± 0.0222 |

All four result tables (label efficiency at 200/500/1,000/2,000 labels, number of unlabeled clips,
comparison table, gain of the amalgamation step) are in [`results/`](results/four_results_test.md);
in those generated files the proposed model appears under its working name `Mark5.3`.

They can be rebuilt from the stored per-run outputs without audio or training:

```bash
bash scripts/reproduce_results.sh    # rebuilds results/four_results_* and checks they are byte-identical
```

## Repository layout

| Path | Content |
|---|---|
| `model/` | Training, evaluation, teacher scoring, meta-classifier, experiment runner, result tables |
| `vild/`, `shared_vild/`, `utils/` | Model, data parser and configuration code used by `model/` |
| `teacher_mn10/` | Training code of the binary mn10 teachers |
| `third_party/efficientat/` | EfficientAT mn10 model code (MIT License, see its LICENSE) |
| `preprocessing/` | Tools that cut, resample and check the audio clips |
| `reproducibility/` | Candidate lists, gate reports and download tools for the external teacher data |
| `dataset_*.csv`, `data/` | Index files read by the code, clip provenance and data access notes |
| `resources/` | Teacher manifests (the teacher weights themselves are in the release) |
| `notebooks/` | Colab notebooks used for the experiments (Korean comments) |
| `results/` | The four result tables and figures, plus `results/runs/` with the per-run outputs they are built from |
| `experiments/additional_20261114/` | Additional experiments: component comparisons, full-distribution distillation, teacher-label oracle, source-group re-splits, paired confidence intervals |
| `baselines/` | Comparison models: EfficientAT mn10, PANNs CNN6, CED-Tiny |
| `tests/` | Implementation checks |
| `scripts/` | `download_weights.sh`, `reproduce_results.sh` |

## Setup

Python 3.8. The pinned versions below were used for local CPU checks; training ran on Google Colab with an
NVIDIA L4 GPU.

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Weights

[Release v1.0.0](https://github.com/KAMPA-audio/KAMPA/releases/tag/v1.0.0) contains the 40 student
checkpoints behind the result tables (8 settings × seeds 1-5), the 8 binary mn10 teachers and the 16 files
of the segment teachers (B0) that served as the reference in the teacher gate.

```bash
bash scripts/download_weights.sh
```

This places the students in `model/`, the teachers in `resources/`, downloads the EfficientAT mn10 AudioSet
weights (`weights/mn10_as_mAP_471.pt`, EfficientAT release v0.0.1) and checks every file's SHA-256.

## Data

The audio is not redistributed. [`data/README.md`](data/README.md) lists the source collections, their
licenses, the provenance of every clip and how to rebuild the audio folders.

## Running the experiments

With the audio rebuilt and the weights downloaded. Outside the Colab layout used for the experiments
(`/content/drive/MyDrive/NewProject/codes/experiments/KAMPA`), first point the index files at the audio:

```bash
python scripts/set_data_root.py                                             # audio in <repo>/data_labeled, data_unlabeled, data_val, data_test
python model/make_teacher_scores.py --teacher_kind mn10                    # teacher score and feature caches -> plots/
python model/make_combiner_labels.py --scores plots/teacher_scores_mn10.csv # meta-classifier pseudo-labels per label budget and seed
python model/label_freshness.py                                             # hash chain teacher -> scores -> pseudo-labels
python model/run_experiments.py --mark_version mark5.2 --preset cmb --cmb_paper --seeds 1 2 3 4 5   # 40 student runs
python model/make_four_results.py --baseline_root baselines                 # the four result tables
```

`run_experiments.py` skips a run only when its checkpoint was trained on the current data lists and code,
so the released checkpoints are retrained by this command. To evaluate a released checkpoint instead, call
`model/eval.py` with the run's flags, for example (all labels, 2,000 unlabeled clips, seed 1):

```bash
python model/eval.py --split test --seed 1 --unlabeled_n 2000 --pseudo_source cmb \
    --linear_head --head_pretrained_hidden --hard_weight 1.0 --teacher_seg_mean --teacher_clean_view \
    --bg_weight 0 --student_raw_wave --joint_forward
```

On a CPU this reproduces the stored test predictions of that run for all 430 clips (class probabilities
differ from the GPU run by less than 0.01). For the other checkpoints, map the parts of the file name to
flags: `_l200` → `--labeled_n 200` (no `_l` part: all 2,000 labels), `_u500` → `--unlabeled_n 500`,
`_pscmb` / `_psfix` → `--pseudo_source cmb` / `fix` (no `_ps` part: leave the flag out), `_s3` → `--seed 3`.

The comparison models are run from their folders, with the repository root as the audio root:

```bash
cd baselines/Baseline_EfficientAT_mn10
python run_experiments.py --phase main --conds sup --labeled_ns 0 1000 500 200 --seeds 1 2 3 4 5 --data_root ../..   # 0 = all labels
```

## Tests

```bash
python model/verify_release.py --source_only    # code files against source_manifest.json
```

After `scripts/download_weights.sh`, `python model/verify_release.py` (code, index files and all weights) and
`python tests/verify_release_contract.py` also pass, as long as the index files still carry the original prefix
(run them before `scripts/set_data_root.py`, or restore the files with `git checkout`). The other scripts in `tests/` need the audio as well
(for example `python tests/verify_single_classifier.py --source-root . --audio-root .`).

## License

The code in this repository is released under the [PolyForm Noncommercial License 1.0.0](LICENSE).
`third_party/` and `baselines/*/third_party/` keep their own licenses. The audio collections are governed by
their providers' terms (see `data/README.md`); none of them is relicensed here.

## Citation

See [`CITATION.cff`](CITATION.cff).
