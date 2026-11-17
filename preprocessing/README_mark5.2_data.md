> **Mark5.2 (2026-09-15 신설)** — Mark5(2026-08-23 확정본)를 복사해 만든 데이터 재구성판.
> 바뀐 것: media_talking(TV 만) → **talking**(TV + 사람 말소리 50:50), others → **층간소음이 아닌 먼 소리**
> (집 녹음 50% + ESC-50·AI Hub 교통 50%), Mark4.8 긴 파일 3초 정렬, 평가 전용 세트 `data_eval/`(제로샷 6종·경계 B·ESC far_heldout).
> 데이터 재구성 기록은 `preprocessing/rebuild_mark52_data.py` 머리주석과 `preprocessing/data_prep_20260915/결과요약_20260915.txt`.
>
> **2026-09-16 운영점 자동 선택**: `model/eval.py --split val` 이 argmax + direct 격자에서 val Macro F1 최대 규칙을 골라
> `plots/others_threshold_<run_tag>.json`(체크포인트 sha256 포함)에 쓰고, `eval.py`(test)는 그 파일을 읽어 지문이 맞을 때만 쓴다.
> `run_all.py` 는 6a val → 6b test 순서. 고정값으로 보려면 `--others_threshold 0.09` 또는 `argmax`.
>
> **2026-09-17 백본 교체(6절)**: student 인코더가 사전학습 백본 **EfficientAT mn10**(`vild/vild_model.py Mn10AudioEncoder`)이다.
> - 입력: 파서가 mel 세그먼트(teacher 용, 64×101)와 같은 창의 **파형 1.0초(16 kHz)** 를 함께 내고(`record["wave"]`),
>   student 는 파형 → 16k→32k 리샘플 → Kaldi mel 128 → mn10 → 풀링 960 → Dropout → Linear(384). `config.student_input` 이 고른다.
> - 가중치 `weights/mn10_as_mAP_471.pt`(EfficientAT 릴리스 v0.0.1, MIT, 19.7 MB), 코드 `third_party/efficientat/`(원본 그대로, MIT).
>   **torchvision 이 필요하다**(ConvNormActivation). 코랩에는 있고, 로컬 venv 는 `torchvision==0.19.1+cpu`.
> - SpecAugment: student 는 인코더 안에서 128 mel 에 비율(8/64·16/101)·bin 별 시간 평균 채움으로, teacher 는 기존 64 mel 규칙으로.
>   teacher 와 student 의 마스크는 서로 독립이다("같은 view" 가 아님, 09-17 검수 P4).
> - 입력 스케일(09-17 검수 P9): 이 파서(`utils/parser_utils.py load_audio_file`)는 클립마다 **피크 정규화**(|x|max=1)를 하고, 비교모델
>   4폴더(`common/data.py load_clip`)는 원 진폭 그대로 넣는다. 같은 mn10 백본이라도 "같은 전처리" 는 아니므로 논문에 그렇게 쓰지 않는다.
> - 학습률: 백본 5e-5(`config.encoder_learning_rate`), 새 층(투영 960→384·branch·background) 1e-4. 전 층 파인튜닝. (09-17 검수: 투영층이 인코더 그룹에 섞여 5e-5 로 들어가던 것을 새 층 그룹으로 옮김)
> - 나머지(세그먼트 5개·teacher 합의·두 갈래 헤드·텍스트 앵커·others 규칙·feature KD)는 Mark5 그대로. `encoder_type="cnn"` 으로 두면 옛 3층 CNN 경로.
> - 기본 `--mark_version` 은 mark5.2. 로컬 smoke(2026-09-17, 27클립·2에폭·CPU): 학습 → val 자동 선택 → test → profile_efficiency 전부 통과.
> 아래는 Mark5 원본 README 그대로.

# Mark5.0
- Based on Mark3.2 but much better with vision-like perspectives

# License
- This project is licensed under the PolyForm Noncommercial License 1.0.0.  
- Commercial use is not permitted.
- See the [LICENSE](./LICENSE) file for details.


