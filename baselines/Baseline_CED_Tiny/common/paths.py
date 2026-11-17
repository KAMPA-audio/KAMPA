# -*- coding: utf-8 -*-
"""경로 해석 — dataset_*.csv 의 코랩 드라이브 경로를 지금 실행 중인 환경의 파일로 되돌린다.

[신설 2026-09-05]

dataset_labeled.csv 등 네 인덱스는 Mark5_EXP 에서 그대로 복사해 왔고, 안에 적힌 경로는
코랩 드라이브 경로다.
    /content/drive/MyDrive/NewProject/codes/experiments/Mark5.2/data_labeled/construction/xxx.wav
    (2026-09-08 이전 드라이브 이름은 AImodels, 옛 인덱스는 Mark5_EXP 였다)

이 경로가 그대로 있으면(코랩) 그것을 쓰고, 없으면(로컬) 'Mark5_EXP/' 뒤의 꼬리를
data_root 아래에서 찾는다.

[변경 2026-11-17] 데이터 창고 Mark5.2 를 모델 저장소 KAMPA(구 Mark5.3)로 옮겼다. 인덱스 경로는 'KAMPA/' 뒤가 꼬리이고
옆 폴더 탐색도 KAMPA 를 먼저 본다. 옛 Mark5.2·Mark5_EXP 경로도 그대로 풀린다.

[변경 2026-09-16] 새 데이터는 Mark5.2(2026-09-15 재구성판)에서 온다. 인덱스 경로 꼬리는 'Mark5.2/' 뒤이고
옆 폴더 탐색도 Mark5.2 를 먼저 본다(없으면 Mark5_EXP). 클래스도 media_talking -> talking 으로 바뀌었다
(Mark5.2 vild/vild_config.py 의 mark5.2 분기와 같은 순서). 옛 Mark5_EXP 인덱스(media_talking 라벨)는
이 클래스 목록과 맞지 않으므로 새 코드로는 돌리지 않는다. 로컬 Mark5_EXP 에는 전처리 전 폴더(data_source_*)만 있으므로
data_labeled -> data_source_labeled 별칭도 함께 시도한다. 전처리(fix_audio_length.py)는
3초 미만 클립에 무음을 붙이는 것뿐이고 4,860개 전부 3.0초 이상이라(2026-09-05 헤더 전수
확인) 두 폴더의 소리는 같다.

오디오를 이 폴더에 복사하지 않는 이유: 894MB 를 세 폴더에 복사하면 2.7GB 인데 C 드라이브가
99% 라서(2026-09-05 df 실측 3.4GB 남음) Mark5_EXP 것을 가리켜 쓴다. 드라이브에서도 같은
원리로 Mark5_EXP 폴더의 데이터를 그대로 읽으므로 다시 올릴 필요가 없다.
"""
import os
import re

# Mark5.2 와 같은 9 클래스, 같은 순서(Mark5.2 vild/vild_config.py mark5.2 분기). 출력 인덱스가 그대로 이 순서다.
# [변경 2026-09-16] media_talking(TV 만) -> talking(TV + 사람 말소리 50:50). 데이터 재구성(2026-09-15)에 맞춤.
CLASS_NAMES = [
    "heavy_impact", "dragging", "construction", "machine_noise",
    # [2026-11-18] talking → speech(논문 Speech/말소리, 제안모델 KAMPA 와 같음). 정의와 순서는 그대로다.
    "speech", "water_toilet", "water_shower", "dog_bark",
    "others",
]
LABEL_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
# [2026-11-18] 오디오 폴더·파일 이름(talking/)과 끝난 회차의 결과 파일(합의·상한 라벨 CSV, 예측 CSV 의 Prob_talking)은 옛 이름 그대로다.
#   라벨은 LABEL_MAP 이 옛 이름도 같은 번호로 받고, 예측 CSV 의 열 이름은 canonical_row 로 맞춘다.
LEGACY_CLASS_NAMES = {"talking": "speech"}
LABEL_MAP.update({old: LABEL_MAP[new] for old, new in LEGACY_CLASS_NAMES.items()})
_LABEL_FIELDS = ("label", "True Label", "Predicted Label", "Raw Predicted Label")
_COLUMN_PREFIXES = ("PreBias_Prob_", "Prob_")


def canonical_column(name):
    """Prob_talking 같은 클래스 열 이름을 새 이름(Prob_speech)으로. 다른 열은 그대로."""
    for prefix in _COLUMN_PREFIXES:
        if name.startswith(prefix):
            return prefix + LEGACY_CLASS_NAMES.get(name[len(prefix):], name[len(prefix):])
    return name


def canonical_row(row):
    """CSV 한 행(dict)의 클래스 열 이름과 라벨 값(True/Predicted Label 등)을 새 이름으로 바꾼 새 dict."""
    return {canonical_column(k): (LEGACY_CLASS_NAMES.get(v, v) if k in _LABEL_FIELDS else v) for k, v in row.items()}

_TAIL_RE = re.compile(r"(?:KAMPA|Mark5(?:\.2|_EXP))[\\/](.+)$")   # [변경 2026-11-17] KAMPA(데이터 이전) 우선, 옛 Mark5.2·Mark5_EXP 경로도 풀린다
_SOURCE_ALIAS = {
    "data_labeled": "data_source_labeled",
    "data_unlabeled": "data_source_unlabeled",
    "data_val": "data_source_val",
    "data_test": "data_source_test",
}


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def default_data_root():
    """오디오를 찾을 기준 폴더. 환경변수 > 옆의 KAMPA > 옆의 Mark5.2 > 옆의 Mark5_EXP > 이 폴더 순. [변경 2026-11-17] KAMPA 우선"""
    env = os.environ.get("BASELINE_DATA_ROOT")
    if env:
        return env
    for name in ("KAMPA", "Mark5.2", "Mark5_EXP"):
        sibling = os.path.abspath(os.path.join(project_root(), "..", name))
        if os.path.isdir(sibling):
            return sibling
    return project_root()


def resolve_audio_path(path, data_root):
    """CSV 의 경로 하나를 실제 파일 경로로. 못 찾으면 None."""
    p = str(path).replace("\\", "/")
    if os.path.isfile(p):
        return p
    m = _TAIL_RE.search(p)
    tail = m.group(1) if m else os.path.basename(p)
    parts = [x for x in tail.split("/") if x]
    candidates = [os.path.join(data_root, *parts), os.path.join(project_root(), *parts)]
    if parts and parts[0] in _SOURCE_ALIAS:
        alias = [_SOURCE_ALIAS[parts[0]]] + parts[1:]
        candidates += [os.path.join(data_root, *alias), os.path.join(project_root(), *alias)]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def resolve_csv(filename):
    """이 폴더 루트의 인덱스 CSV. 없으면 바로 멈춘다(조용히 다른 데이터로 돌지 않게)."""
    path = os.path.join(project_root(), filename)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"[ERROR] 인덱스 CSV 가 없습니다: {path} — KAMPA 에서 복사해 두십시오.")


def index_fingerprint(filenames=("dataset_labeled.csv", "dataset_val.csv", "dataset_unlabeled.csv", "consensus_labels.csv", "ceiling_labels.csv")):
    """[2026-09-17 / 2차 검수 C-3] 학습에 쓴 인덱스 CSV 의 sha1 — train_meta 에 적고 run_experiments 가 재개 판정 때 대조한다.
    데이터 판(09-06 옛 데이터 / 09-15 재구성판)이 태그에 없어서, 옛 산출물이 남아 있으면 --force 없이 이어 돌릴 때 [건너뜀] 으로
    새 표에 섞일 수 있었다. 파일이 없으면 None(그 파일이 나중에 생기면 지문이 달라져 재학습 — 문서 순서(셀 3 복사 → 셀 7)에서는 안 생긴다).
    [3차 검수 P-2] unlabeled·consensus·ceiling CSV 도 넣는다 — 3부 [셀 7·7b] 를 다시 돌려 라벨이 바뀌면 pl/ceil 회차가 다시 돈다."""
    import hashlib
    out = {}
    for fn in filenames:
        path = os.path.join(project_root(), fn)
        if not os.path.isfile(path):
            out[fn] = None
            continue
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[fn] = h.hexdigest()
    return out
