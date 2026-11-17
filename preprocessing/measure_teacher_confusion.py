#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_teacher_confusion.py — 교사 k 가 다른 타겟 클래스 j 의 소리에 얼마나 손을 드는지(8×8)를 교사 쪽 val 에서 잰다 (2026-10-23).

왜: 설계 문서 'Mark5.3 개선 설계' 교사 개선판(B+) — others 의 "다른 타겟" 자리(train 250 · val/test 107)를 7종에 균등하게
    나누지 않고 실제로 헷갈리는 클래스 쪽에 더 배분한다. 배분 비율은 Mark5.3 데이터가 아니라 교사 쪽 val 에서 잰다(학생 평가 자료를 보지 않기 위해).
무엇을: 3차 교사(Mark4.k/model/best_teacher_*_mark4.k.pth, 조각 교사) 8명을 증강 없이 eval 모드로, 교사 j 의 val 양성 215클립
    (Mark4.j/data/val/<클래스 j>_val_*.wav) 전부에 돌린다. 클립 점수 = 조각 5개의 자기 클래스 확률 평균(Mark5.3 --teacher_seg_mean 과 같은 집계).
    손들기 = 클립 점수 ≥ 0.5.
산출물: rebuild_cache/teacher_confusion_<날짜>.json — raise_rate[k][j] · mean_score[k][j] · 교사 파일 SHA-256 · 클립 수.
사용: python preprocessing/measure_teacher_confusion.py
"""
import os, sys, csv, json, time, hashlib, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                   # noqa: E402

CLASSES = ["heavy_impact", "dragging", "construction", "machine_noise", "talking", "water_toilet", "water_shower", "dog_bark"]
VER_OF = {v: k for k, v in R.EXPERT_TARGET_CLASS.items()}
COLAB = "/content/drive/MyDrive/NewProject/codes/experiments/"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    code = os.path.join(R.EXPERT_ROOT, "Mark4.1")                  # Mark4.x 코드는 8곳 같다 — 모델 정의·조각 파서만 빌려 쓴다
    for p in (code, f"{code}/model", f"{code}/utils", f"{code}/vild", f"{code}/shared_vild"):
        sys.path.insert(0, p)
    import torch
    from vild_config import AudioViLDConfig
    from vild_model import ViLDTextHead, build_teacher_encoder
    from vild_parser_teacher import AudioParser
    from checkpoint_utils import load_checkpoint, resolve_state_dict
    torch.set_num_threads(4)

    # 교사 j 의 val 양성 클립 → 조각 텐서 [n, 5, C, F, T]
    clips, files = {}, {}
    for cls in CLASSES:
        ver = VER_OF[cls]; repo = os.path.join(R.EXPERT_ROOT, "Mark" + ver[4:])
        cfg = AudioViLDConfig(mark_version=ver); cfg.device = "cpu"
        parser = AudioParser(cfg, segment_mode=True)
        with open(os.path.join(repo, "preprocessing", f"dataset_index_{ver}.csv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["label"] == cls and f"/data/{a.split}/" in r["path"].replace("\\", "/")]
        segs = []
        for r in rows:
            p = r["path"].replace("\\", "/").replace(COLAB + "Mark" + ver[4:], repo)
            s = [x for x in parser.load_and_segment(p) if isinstance(x, torch.Tensor)]
            assert len(s) == 5, (p, len(s))
            segs.append(torch.stack(s))
        clips[cls] = torch.stack(segs)
        print(f"[DATA] {cls:14s} {a.split} 양성 {len(rows)}클립 ({time.time()-t0:.0f}s)", flush=True)

    raise_rate, mean_score = {}, {}
    for k in CLASSES:
        ver = VER_OF[k]; repo = os.path.join(R.EXPERT_ROOT, "Mark" + ver[4:])
        cfg = AudioViLDConfig(mark_version=ver); cfg.device = "cpu"
        encp, clsp = (os.path.join(repo, "model", f"best_teacher_{x}_{ver}.pth") for x in ("encoder", "classifier"))
        e, c = load_checkpoint(encp, map_location="cpu"), load_checkpoint(clsp, map_location="cpu")
        enc = build_teacher_encoder(cfg); enc.load_state_dict(resolve_state_dict(e, "model_state_dict")); enc.eval()
        head = ViLDTextHead(cfg); head.load_state_dict(resolve_state_dict(c, "classifier_state_dict")); head.eval()
        te = c["text_embeddings"]
        files[k] = {"encoder": os.path.relpath(encp, R.CODES_ROOT), "encoder_sha256": sha(encp),
                    "classifier": os.path.relpath(clsp, R.CODES_ROOT), "classifier_sha256": sha(clsp)}
        raise_rate[k], mean_score[k] = {}, {}
        for j in CLASSES:
            X = clips[j]; ps = []
            with torch.no_grad():
                for b in range(0, len(X), 32):
                    xb = X[b:b + 32]
                    p = torch.softmax(head(enc(xb.flatten(0, 1)), te), -1)[:, 0].reshape(len(xb), 5).mean(1)
                    ps.append(p)
            p = torch.cat(ps)
            raise_rate[k][j] = round(float((p >= a.threshold).float().mean()), 4)
            mean_score[k][j] = round(float(p.mean()), 4)
        print(f"[TEACHER] {k:14s} " + " ".join(f"{j[:6]}={raise_rate[k][j]:.3f}" for j in CLASSES), flush=True)

    out = a.out or os.path.join(R.CACHE_DIR, f"teacher_confusion_{time.strftime('%Y%m%d')}.json")
    json.dump({"split": a.split, "threshold": a.threshold, "classes": CLASSES, "n_clips": {c: len(clips[c]) for c in CLASSES},
               "raise_rate": raise_rate, "mean_score": mean_score, "teacher_files": files,
               "teachers": "3차(조각 교사, Mark4.k/model, 2026-09-23)", "clip_score": "조각 5개 자기 클래스 확률 평균",
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[완료] {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
