# -*- coding: utf-8 -*-
"""Mark5.3 검증 결합기의 한 라벨 가짜 정답을 비교모델이 그대로 읽는다(조건 plc).

[신설 2026-10-22 / 검증 결합기] 설계 문서 '비교모델 조건': 비교모델은 Mark5.3 과 같은 파일을 읽고, 두 후보 라벨은 받지 않는다.
복사본을 이 폴더에 두지 않는다 — 옆 저장소 Mark5.3 의 파일을 직접 읽어야 교사·결합기를 다시 돌렸을 때 옛 복사본이 섞이지 않는다
(2026-10-22 수진님 지시: "항상 최신 것을 쓰도록").

    파일   : <Mark5.3>/pseudo_labels_cmbone_l<n|all>_s<seed>.csv   (make_combiner_labels.py 산출, path,label,confidence)
    최신 판: Mark5.3 의 model/label_freshness.check_pseudo_csv — 결합기 json 의 CSV 해시 · 점수 캐시 해시 · resources/ 교사 해시를 대조
"""
import hashlib
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK53_ROOT = os.environ.get("MARK53_ROOT") or os.path.join(os.path.dirname(ROOT), "KAMPA")   # [변경 2026-11-17] 저장소 이름 Mark5.3 → KAMPA


def pseudo_csv_path(labeled_n, seed):
    """정답 라벨 수(None = 전량)·seed 의 결합기 한 라벨 파일. 결합기는 그 조건의 정답 라벨로 학습돼 정답 수마다 파일이 다르다."""
    lname = "all" if labeled_n is None else str(int(labeled_n))
    return os.path.join(MARK53_ROOT, f"pseudo_labels_cmbone_l{lname}_s{int(seed)}.csv")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _freshness_module():
    path = os.path.join(MARK53_ROOT, "model", "label_freshness.py")
    if not os.path.isfile(path):
        raise SystemExit(f"[ERROR] KAMPA(구 Mark5.3) 저장소를 찾지 못했습니다(최신 판 확인 모듈 없음): {path}\n"
                         "        비교모델 폴더 옆에 KAMPA 가 있어야 합니다(환경 변수 MARK53_ROOT 로 바꿀 수 있음).")
    spec = importlib.util.spec_from_file_location("mark53_label_freshness", path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    for p in (os.path.join(MARK53_ROOT, "model"), os.path.join(MARK53_ROOT, "vild"), os.path.join(MARK53_ROOT, "utils"),
              os.path.join(MARK53_ROOT, "shared_vild"), MARK53_ROOT):
        if p not in sys.path:
            sys.path.append(p)      # 조각 교사(B0) 이름을 풀 때 Mark5.3 의 train_mark5 를 읽는다
    spec.loader.exec_module(mod)
    return mod


def check_fresh(path):
    """그 파일이 지금 Mark5.3 교사·점수 캐시·결합기의 최신 판인지 대조하고, 맞으면 파일 해시를 돌려준다. 어긋나면 멈춘다."""
    mod = _freshness_module()
    try:
        meta = mod.check_pseudo_csv(path)
    except mod.StaleFileError as e:
        raise SystemExit(str(e) + "\n        KAMPA 에서 make_teacher_scores.py → make_combiner_labels.py 를 지금 교사로 다시 돌리십시오.")
    if meta is None:
        raise SystemExit(f"[ERROR] 결합기 가짜 정답 파일 이름이 아닙니다: {path}")
    return sha256_file(path)
