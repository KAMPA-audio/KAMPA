# Data

No audio is distributed with this repository. The recordings come from third-party collections whose
terms we follow; obtain them from the original providers and rebuild the clips with the tools listed below.

## Student collection (4,860 three-second, 16 kHz mono clips)

| Source | Clips | Original repository | License / terms |
|---|---:|---|---|
| AI Hub living-environment noise (dataset 71296) | 3,847 | <https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71296> | Provider usage policy and approval (<https://aihub.or.kr/intrcn/guid/usagepolicy.do>). Audio is not redistributed; request access from AI Hub. |
| SINS / DCASE 2018 Task 5, development set | 520 | <https://zenodo.org/records/1247090> | CC BY-NC 4.0 |
| CHiME-Home | 287 | <https://archive.org/details/chime-home> | CC BY-NC-SA 3.0 |
| ESC-50 | 206 | <https://github.com/karolpiczak/ESC-50> | CC BY-NC 3.0 for the collection; keep the per-file Freesound attribution in its LICENSE |

Splits: 2,000 labeled training clips, 2,000 clips whose labels are withheld during training (unlabeled),
430 validation clips and 430 test clips.

The eight binary teachers were trained on other collections as well (SINS evaluation set, CHiME-6, FSD50K,
Clotho, BeDAViN, SNU-B36-50E, BBC RemArc Sound Effects, VOiCES). Their candidate lists, gate reports and
download tools are in [`reproducibility/`](../reproducibility/README.md).

## Files

| File | Content |
|---|---|
| `dataset_labeled.csv`, `dataset_unlabeled.csv`, `dataset_val.csv`, `dataset_test.csv` (repository root) | The index files the code reads: `path,label` (`path` only for unlabeled). |
| `dataset_eval_{boundary_B,far_heldout,zeroshot6}.csv` (repository root) | Additional evaluation sets used by the open-set and zero-shot scripts. |
| `data/provenance_student.csv` | One row per student clip: `filename, source, recording, split, label, group, sha256`. `recording` is the original file ID in the source collection (for AI Hub, the AI Hub file name), `group` is the recording group used to keep splits apart, and `sha256` is the hash of the rebuilt clip. |

Paths in the index files start with the Colab location used for the experiments
(`/content/drive/MyDrive/NewProject/codes/experiments/KAMPA/`), and the training and evaluation code opens
them as written. Either keep the repository at that location on Google Colab, or rebuild the audio as
`<root>/data_labeled/<class>/<file>.wav`, `<root>/data_unlabeled/<file>.wav`, `<root>/data_val/...`,
`<root>/data_test/...` and run `python scripts/set_data_root.py --root <root>` (default root: this
repository) to rewrite the prefix in the index files.

### Class names

| Class | Paper name |
|---|---|
| `heavy_impact` | Heavy impact |
| `dragging` | Dragging |
| `construction` | Construction |
| `machine_noise` | Machine noise |
| `speech` | Speech (adult speech and TV, child speech excluded) |
| `water_toilet` | Toilet flushing |
| `water_shower` | Shower sounds |
| `dog_bark` | Dog barking |
| `others` | Other sounds |

The class `speech` was called `talking` during the experiments. Audio folders and file names keep the old
name (`data_labeled/talking/talking_train_001.wav`), and so do stored experiment outputs such as
`Prob_talking` columns; `model/class_names.py` maps the old name when such files are read.

## Rebuilding the audio

1. Obtain the source collections listed above.
2. The scripts in `preprocessing/` cut and resample the clips (`mark5_slicer.py`, `esc50_slicer.py`,
   `rebuild_mark52_data.py`, `remove_child_speech.py`, `fix_audio_length.py`); `preprocessing/README_mark5.2_data.md`
   records how the collection was built. These scripts were written for our local folder layout, so check
   the source paths at the top of each script before running it.
3. Check each rebuilt clip against `sha256` in `data/provenance_student.csv`.

The provenance file is enough to locate every original recording, but it does not record every crop
offset; where a script picks the window from the signal (for example the loudest 3 s), the same script
must be used to obtain an identical clip.
