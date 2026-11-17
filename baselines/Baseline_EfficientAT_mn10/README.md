# EfficientAT mn10 — Mark5 비교실험 베이스라인

EfficientAT mn10 (Schmid et al., ICASSP 2023) — AudioSet 사전학습 MobileNetV3(폭 1.0) 4.88M 파라미터, 32 kHz · 128 mel. 헤드는 저자 방식(get_model num_classes=9, mlp).

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
- [2026-10-25~ A안] 옵티마이저 = Adam, 학습률은 아래 선정 기록의 값 고정(`common/naming.py` `adam_lr`: sup·pl·plc·ceil 1e-4) — 다시 고르지 않음, run_tag `_adam`. Mark5.3(제안모델)과 같은 절차
- (2026-10-22~10-25) 자동 학습률 Prodigy(run_tag `_pd`)를 썼다가 걷어냈다. 학습 초반 보폭이 Adam 1e-4 의 13~60배로 커져 사전학습 가중치의 60.7% 가 0 근처로 줄었고(정답 2,000), test Acc 가 Adam 보다 6~9%p 낮았다
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
run_tag = `<model>_<cond>_adam[_l<n>][_u<n>]_s<seed>[_smoke]` (규칙은 `common/naming.py` 한 곳, 예: `efficientat_mn10_sup_adam_s1`. 2026-10-21까지는 `_lr<lr>`, 10-22~10-25 Prodigy 는 `_pd`)

## 표준 반지도 기준선 — 논문 표 1·표 3 (2026-10-07 신설)
같은 mn10 에서 표준 반지도 방법 4개를 다시 구현해 Mark5.3 과 비교한다(FixMatch·FlexMatch·USB·Cances 2022 의 비교 방식, 설계 근거는 paper-wiki `wiki/concepts/ssl-comparison-protocol.md`).

| 조건(`--cond`) | 라벨 없는 클립에 거는 손실 (USB 오디오 설정값) | 평가 모델 |
|---|---|---|
| `pseudolabel` | 증강 없는 입력의 예측이 0.95 이상이면 그 클래스로 CE, 가중 1 | 학습 모델 |
| `meanteacher` | 학생(SpecAugment 입력)과 EMA 선생(증강 없는 입력)의 확률 MSE, 가중 50, EMA 0.99(USB 0.999, 아래 10-08) | EMA 선생 |
| `fixmatch` | 증강 없는 입력 예측 0.95 이상 → 그 클래스로 SpecAugment 입력을 CE, 가중 1 | 학습 모델 |
| `freematch` | FixMatch 와 같되 문턱을 전체·클래스별로 스스로 조정(EMA 0.99, USB 0.999) + 공정성 손실 0.001 | 학습 모델 |

- 배치 = 정답 클립 + 라벨 없는 클립 2,000개를 섞어 16(Mark5.3 과 같은 구성). 비지도 손실은 처음 2에폭 선형 램프업.
  학습 모델의 순전파는 네 방법 모두 섞은 배치 16개 전체를 한 번에 한다(BN 이 16개로 계산된다). 옵티마이저·조기 종료·80에폭·patience 10 은 sup 과 같다(`common/ssl_loop.py`, 손실은 `common/ssl.py`).
- **10-08 수정(구현 판 `ssl_v20261008`, Pseudo-Label·Mean Teacher·FreeMatch):** ① USB 의 PL·MT 는 정답 배치(8개)를 따로 순전파하는데, 섞은 배치에서 그대로 하면
  정답 라벨 200 일 때 BN 이 정답 1~2개로 계산됐다(10-07 판에서 두 방법만 하한보다 낮았던 원인) → 배치 전체 한 번 순전파로 고침.
  ② USB 의 EMA 0.999 는 102,400 걸음 기준이라 조기 종료까지 수천 걸음인 우리 학습에서는 선생·문턱이 크게 뒤처진다 → Mark5.3 FM 부품과 같은 규칙으로 0.99.
  FixMatch 는 바뀐 것이 없어 `ssl_v20261007` 결과를 그대로 쓴다. 10-07 판 결과는 드라이브 각 폴더의 `_ssl_v20261007/` 에 보관.
- [2026-10-25~ A안] 표준 반지도 4방법도 Adam, 학습률은 방법별 선정 기록의 값 고정(`adam_lr`: Pseudo-Label·Mean Teacher·FixMatch 1e-4, FreeMatch 5e-5), run_tag `_adam`. (10-22~10-25 Prodigy `_pd` 는 걷어냈다.)
- (2026-10-21까지) 학습률은 방법마다 후보 3개(1e-4·5e-5·2e-5)를 seed 0·labeled 전량에서 val Macro F1 로 고른다 → `plots/lr_selected_<방법>.json`(sup 의 `lr_selected.json` 은 그대로, 2026-10-08 선정).
- 숨긴 정답 `ceiling_labels.csv` 는 가짜 정답 품질·양 기록(표 3)에만 쓴다 — `plots/pl_quality_<run_tag>.csv`(에폭별) · `train_meta` 의 `pl_quality_final`(학습 뒤 가장 좋은 체크포인트로 2,000개 전부).
- 재개 판정: train_meta 가 있어야 하고 `ssl_impl_version`(= `common/naming.py` `ssl_impl_version(방법)`, 방법별 판)이 같아야 완료다. 표 생성기도 지금 판 결과만 넣는다.

```
python run_experiments.py --preset ssl --data_root <KAMPA 폴더>      # 방법마다 선정 3회 + labeled 전량·1,000·500·200 x seed 1~5 = 92회
python make_ssl_table.py --mark53_dir ../KAMPA/plots                 # 표 1(plots/ssl_table1_test.csv) · 표 3(plots/ssl_table3_pl_quality.csv)
python count_expert_data.py --hash                                      # 표 1 "추가 지식" 열: 전문가 8개 학습 데이터 개수(plots/expert_data_counts.*)
```
`reselect_others_threshold.py` 는 sup·pl·ceil 태그만 읽는다(표준 반지도 회차의 운영점은 eval.py 가 val 에서 같은 규칙으로 고른다).

## 출처·라이선스
- `third_party/efficientat/`: fschmid56/EfficientAT 의 모델·전처리 파일 원본 사본(MIT). 세부는 그 폴더의 SOURCE.md
- `weights/mn10_as_mAP_471.pt`: 저장소 릴리스 v0.0.1 (19.7 MB)
- 추가 패키지: torchvision (로컬 venv 에는 이미 있음. ★코랩에서는 설치하지 말 것 — 기본 설치본을 그대로 쓴다)
