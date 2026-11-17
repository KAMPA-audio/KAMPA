# PANNs CNN6 — Mark5 비교실험 베이스라인

PANNs CNN6 (Kong et al., IEEE/ACM TASLP 2020) — AudioSet 사전학습 4.8M 파라미터 CNN, 32 kHz · 64 mel. 헤드는 저장소 finetune_template 방식(임베딩 512 뒤 새 fc).

이 폴더는 Mark5(제안 모델)와 같은 표에 놓을 비교모델 하나를 같은 데이터·같은 분할·같은 학습 골격으로
파인튜닝하는 코드다. 조건의 근거와 결정 내역은 **`Downloads/Mark5_비교실험_조건표_20260905.pdf`** (2026-09-05).
같은 구조의 폴더가 셋 있다: `Baseline_PANNs_CNN6` · `Baseline_EfficientAT_mn10` · `Baseline_CED_Tiny`.
`common/`·`train.py`·`eval.py`·`run_experiments.py`·`aggregate_baseline.py` 는 세 폴더가 똑같은 사본이고,
모델마다 다른 것은 `model_def.py`·`model_info.py`·`third_party/`·`weights/` 뿐이다.

## 실험 구성 (조건표 요약)
- 결과표 세 블록: **A** labeled 2,000 지도학습(`--cond sup`, 바닥) / **B** labeled 2,000 + 우리 합의 라벨(`--cond pl`, 반지도) / **C** labeled 2,000 + unlabeled 2,000 에 진짜 라벨(`--cond ceil`, 천장 — 2026-09-06 추가, Cances et al. 2022 의 Supervised 100% 기준선. 제안 방법이 아니라 상한을 재는 조건)
- 공통: Adam · weight decay 1e-4 · batch 16 · 최대 80 epoch · ReduceLROnPlateau(0.5, 3) · early stopping 10 ·
  best-val 체크포인트 · seed 1~5 · SpecAugment(Mark5 와 같은 비율: 주파수 12.5%×2, 시간 15.8%×2) · CE(hard label)
  · 전 층 파인튜닝 · 지표 6종은 Mark5_EXP eval.py 와 같은 계산·같은 CSV 형식
- [2026-10-25~ A안] 옵티마이저 = Adam, 학습률은 아래 선정 기록의 값 고정(`common/naming.py` `adam_lr`: 1e-4) — 다시 고르지 않음, run_tag `_adam`. Mark5.3(제안모델)과 같은 절차
- (2026-10-22~10-25) 자동 학습률 Prodigy(run_tag `_pd`)로 바꿨다가 걷어냈다(같은 날 EfficientAT·CED-Tiny 에서 사전학습 가중치가 지워짐). PANNs 의 Prodigy 회차는 코랩 노트북에 torchlibrosa 설치 셀이 빠져 한 번도 돌지 않았다
- (2026-10-21까지) 학습률만 모델마다 val Macro F1 로 {1e-4, 5e-5, 2e-5} 중 선택(`run_experiments.py --phase sweep` → `plots/lr_selected.json`, 2026-09-24 선정 1e-4), 본실험은 새로 seed 1~5
- 판정 규칙: [2026-10-26] 보정 없는 argmax(Mark5.3 과 같음, `common/others_rule.DECISION_POLICY = "argmax_v1"`). `eval.py --split val` 이 이 규칙과 체크포인트 지문을 `plots/others_threshold_<run_tag>.json` 에 적고 test·eval_* 가 읽는다(sha256 대조, 없거나 옛 규칙이면 멈춤). 실행기는 옛 val 선택 json(2026-09-17~10-26 의 클래스 편향 + others 운영점)인 회차를 '안 끝남' 으로 보고 val·test 만 다시 평가한다(학습은 그대로). `make_argmax_summary.py` 가 회차별 argmax 4지표 표를 만든다. `reselect_others_threshold.py` 는 09-07 옛 산출물 재처리용으로만 남김.

## 데이터
- 진폭(09-17 검수 P9): `common/data.py load_clip` 은 파형을 원 진폭 그대로 넣는다. Mark5.2 파서는 클립마다 피크 정규화를 하므로, 논문에 "같은 전처리" 라고 쓰지 않는다.
- `dataset_{labeled,unlabeled,val,test}.csv` 는 Mark5_EXP 에서 그대로 복사한 것(분할이 자동으로 같아진다).
- 오디오는 이 폴더에 복사하지 않는다. CSV 의 드라이브 경로가 그대로 있으면(코랩) 그것을 읽고, 없으면(로컬)
  `--data_root`(기본: 옆의 `../Mark5_EXP`) 아래에서 같은 꼬리 경로를 찾는다(`common/paths.py`).
  코랩에서는 `--data_root /content/drive/MyDrive/NewProject/codes/experiments/KAMPA` 로 KAMPA 구글 드라이브 폴더를 쓴다(2026-11-17 데이터를 Mark5.2 에서 KAMPA 로 옮김)
  (2026-09-16 갱신: 구글 드라이브 코드 폴더는 2026-09-08 부터 `NewProject/`, 데이터는 2026-09-15 재구성판 Mark5.2. 옛 실험은 `AImodels/…/Mark5_EXP` 였다).
- `consensus_labels.csv` (B 블록용)는 Mark5_EXP 에서 `python model/make_consensus_labels.py` 로 만든 뒤 이 폴더에 복사한다.

## 실행
```
python run_experiments.py --dry_run                      # 계획만
python run_experiments.py                                # 본실험(sup·pl·ceil × seed 1~5 = 15회). 학습률 선정 없음(선정값 고정)
# (2026-10-21까지) --phase sweep 과 --lr 5e-5 는 이제 멈춘다(학습률은 common/naming.py adam_lr 고정)
python aggregate_baseline.py --split test                # 평균±SD, Mark5_EXP 표 1 용 행(plots/paper_rows_*.csv)
python train.py --cond sup --seed 1 --max_clips 16 --epochs 1   # 동작 확인용 -> 태그에 _smoke 가 붙어 본실험과 안 겹침
```
끝난 단계(체크포인트+완주 표식, 지표 6개가 든 요약 CSV)는 다시 실행해도 건너뛴다. 세션이 끊기면 같은 명령을 다시 돌리면 된다.
단, train_meta 에 축소 실행·epoch 축소·증강 off 가 적혀 있는 산출물은 완료로 보지 않고 학습부터 다시 돈다.
오디오를 하나라도 못 찾으면 즉시 멈춘다(`--allow_missing_audio` 는 비상용). `--cond pl` 은 합의 라벨이 0행이면 멈춘다. `--cond ceil` 은 `ceiling_labels.csv`(Mark5_EXP `model/make_ceiling_labels.py` 산출)가 dataset_unlabeled.csv 전부를 덮지 않으면 멈춘다.

## 2026-10-23 판: 검증 결합기 조건 (`--preset cmb`, 8회)
Mark5.3 의 검증 결합기 설계와 짝을 이루는 조건이다. 지도만(`--cond sup`)과 결합기 한 라벨(`--cond plc`)을 정답 전량·1,000·500·200 에서
seed 1 로 한 번씩 돌린다(`CMB_PLAN`).
```
python run_experiments.py --preset cmb --data_root <KAMPA 폴더>   # sup 4 + plc 4 = 8회
python aggregate_baseline.py --split test                          # plc 행 이름: "<모델> + combiner pseudo labels"
```
- `plc` 는 옆 저장소 KAMPA(구 Mark5.3) 의 `pseudo_labels_cmbone_l<n|all>_s<seed>.csv`(한 라벨 행만)를 복사하지 않고 직접 읽는다(`common/mark53_labels.py`,
  위치는 환경 변수 `MARK53_ROOT` 로 바꿀 수 있다). 두 후보 라벨은 받지 않는다.
- 학습 전에 KAMPA 의 `model/label_freshness.check_pseudo_csv` 로 결합기 json · 점수 캐시 · 교사 파일 해시를 대조하고, 어긋나면 멈춘다.
  쓴 CSV 의 SHA-256 은 `train_meta` 의 `pseudo_csv_sha256` 에 남고, 파일이 바뀌면 그 회차를 완료로 보지 않는다.
- run_tag 예: `<model>_plc_adam_l200_s1`(옛 합의 라벨 `pl` 과 태그가 나뉜다). Mark5.3 쪽 표·짝 검정은 `--main_pseudo_source cmb` 로 이 `plc` 를 짝으로 쓴다.

## 산출물
`model/<run_tag>.pth` · `plots/loss_history_<run_tag>.csv`(완주 표식) · `plots/performance_summary_<run_tag>[_val].csv`
· `plots/prediction_details_*.csv` · 혼동행렬·ROC PNG · `plots/lr_selected.json`(Adam 학습률 선정 기록 — 지금 고정값의 근거) · `plots/paper_rows_<model>_<split>.csv` · `logFiles/`
run_tag = `<model>_<cond>_adam[_l<n>][_u<n>]_s<seed>[_smoke]` (규칙은 `common/naming.py` 한 곳, 예: `panns_cnn6_sup_adam_s1`. 2026-10-21까지는 `_lr<lr>`, 10-22~10-25 Prodigy 는 `_pd`)

## 출처·라이선스
- `third_party/panns/`: qiuqiangkong/audioset_tagging_cnn 의 models.py·pytorch_utils.py 원본 사본(MIT). 세부는 그 폴더의 SOURCE.md
- `weights/Cnn6_mAP=0.343.pth`: Zenodo 3987831 (23.7 MB)
- 추가 패키지: torchlibrosa (pip, MIT)
