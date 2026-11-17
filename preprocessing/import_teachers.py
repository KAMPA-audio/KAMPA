# -*- coding: utf-8 -*-
"""전문가(teacher) 체크포인트 16개 반입 — Mark4.1~4.8 의 best_teacher_{encoder,classifier}_mark4.x.pth 를 resources/ 에 복사하고
sha1·크기·시각·출처·운영점(val 자동 선택 json)을 resources/teacher_manifest.json 에 적는다. (체크리스트 5절
"teacher 체크포인트 16개를 Mark5.2 resources/에 넣고 sha1 기록")

[신설 2026-09-17 / Mark5.2]

    python preprocessing/import_teachers.py                                  # 옆의 ../Mark4_series/Mark4.x (로컬 USB)
    python preprocessing/import_teachers.py --src_root /content/drive/MyDrive/NewProject/codes/experiments   # 코랩: Mark4.x 폴더가 experiments/ 바로 아래
    python preprocessing/import_teachers.py --dry_run                        # 무엇을 어디서 가져올지, 지금 resources/ 와 sha1 이 같은지만 본다
    python preprocessing/import_teachers.py --also ../Mark5_EXP2             # Mark5_EXP2/resources/ 에도 같은 16개 복사
    python preprocessing/import_teachers.py --run_subdir _run3_20260923      # [2026-09-30 밤 / 가짜 정답 재료] 보관된 회차의 teacher 를 가져온다
                                                                             #   (각 Mark4.x 의 model/<run_subdir>/ · plots/<run_subdir>/ — 예: 3차 조각 판)

검사·기록
  - 16개가 전부 있어야 진행한다(하나라도 없으면 멈춘다 — 옛 teacher 와 새 teacher 가 섞이는 것을 막기 위해).
  - 각 저장소의 plots/target_threshold_mark4.x.json(4절 운영점 자동 선택 산출물)이 있으면 그 운영점·val 4지표를 manifest 에 **참고로**
    같이 적는다. 그 json 의 체크포인트 지문은 student(distilled_student_*)의 것이라 teacher 파일과 대조하지 않는다 — teacher 는
    val loss 최소 에폭의 저장본이고(teacher_train.py), 운영점은 student 평가에서 고른 값이다. 둘의 날짜가 같은 재학습 회차인지는
    mtime 으로 눈으로 확인한다(manifest 에 둘 다 적힌다).
  - 이미 resources/ 에 같은 sha1 파일이 있으면 "동일" 로만 표시하고 덮어쓰지 않는다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RES_DIR = os.path.join(PROJECT_ROOT, "resources")
MARKS = [f"mark4.{i}" for i in range(1, 9)]
KINDS = ("encoder", "classifier")


def _hash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_src_dir(src_root, mark):
    """Mark4.x 저장소 폴더: <src_root>/Mark4_series/Mark4.x 또는 <src_root>/Mark4.x (코랩 experiments/)."""
    tag = "Mark" + mark[len("mark"):]
    for cand in (os.path.join(src_root, "Mark4_series", tag), os.path.join(src_root, tag)):
        if os.path.isdir(cand):
            return cand
    return None


def main():
    # [2026-11-17] Mark5.2 에서 옮겨 온 옛 반입 도구다(기록으로 보관). 지금 resources/ 의 형식(teacher_manifest.json schema·검증·원자적 교체)은
    # model/import_teachers.py 가 관리하므로, 이 도구가 resources/ 를 덮어쓰지 않게 바로 멈춘다.
    raise SystemExit("[STOP] 옛 반입 도구입니다(Mark5.2 시절 기록). 교사 반입은 model/import_teachers.py 를 쓰십시오 — "
                     "조각 교사(B0): python model/import_teachers.py --source_root ../Mark4_series · "
                     "mn10 교사(B+): python model/import_teachers.py --kind mn10 --source_root ../Mark4_series")
    ap = argparse.ArgumentParser(description="teacher 체크포인트 16개 반입 + manifest")
    ap.add_argument("--src_root", type=str, default=os.path.abspath(os.path.join(PROJECT_ROOT, "..")),
                    help="Mark4.x 저장소들이 있는 상위 폴더(기본: 이 저장소의 부모 = journal/codes)")
    ap.add_argument("--also", type=str, nargs="*", default=[], help="같은 16개를 복사할 다른 저장소 루트(예: ../Mark5_EXP2)")
    ap.add_argument("--run_subdir", type=str, default="",
                    help="[2026-09-30 밤 / 가짜 정답 재료] Mark4.x 의 model/<run_subdir>/ 에서 가져온다(운영점 json 은 plots/<run_subdir>/). "
                         "비우면 model/ 바로 아래(가장 최근 회차). 예: _run3_20260923 = 3차(조각 입력) 보관본")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--allow_threshold_mismatch", action="store_true", help="운영점 json 이 teacher 보다 오래돼도 진행")
    args = ap.parse_args()

    plan, missing = [], []
    for mark in MARKS:
        sdir = _find_src_dir(args.src_root, mark)
        if sdir is None:
            missing.append(f"{mark}: 저장소 폴더 없음 ({args.src_root})"); continue
        for kind in KINDS:
            name = f"best_teacher_{kind}_{mark}.pth"
            src = os.path.join(sdir, "model", args.run_subdir, name)             # run_subdir 가 빈 문자열이면 model/ 바로 아래
            if not os.path.isfile(src):
                missing.append(f"{mark}: {src} 없음"); continue
            plan.append((mark, kind, name, src, os.path.join(sdir, "plots", args.run_subdir, f"target_threshold_{mark}.json")))
    if missing:
        print("[ERROR] 16개가 다 있어야 합니다. 빠진 것:"); [print("   -", m) for m in missing]
        sys.exit(1)

    manifest = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "src_root": os.path.abspath(args.src_root),
                "run_subdir": (args.run_subdir or None), "teachers": {}}
    if args.run_subdir:
        print(f"[INFO] 보관 회차에서 가져온다: 각 Mark4.x 의 model/{args.run_subdir}/ (운영점 json 은 plots/{args.run_subdir}/)")
    problems = []
    for mark, kind, name, src, thr_json in plan:
        sha1, sha256, size = _hash(src, "sha1"), _hash(src, "sha256"), os.path.getsize(src)
        rec = {"file": name, "source": src, "bytes": size, "sha1": sha1, "sha256": sha256,
               "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(src)))}
        if os.path.isfile(thr_json):
            with open(thr_json, encoding="utf-8") as f:
                thr = json.load(f)
            rec["target_threshold_json"] = os.path.relpath(thr_json, args.src_root)
            rec["target_threshold"] = thr.get("threshold")
            rec["student_val_metrics_at_threshold"] = {k: thr.get(k) for k in ("val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1", "val_others_fpr") if k in thr}
            rec["threshold_selected_at"] = thr.get("selected_at")
            if rec["threshold_selected_at"] and rec["threshold_selected_at"][:10] < rec["mtime"][:10]:
                problems.append(f"{name}: 운영점 json({rec['threshold_selected_at']})이 teacher 파일({rec['mtime']})보다 오래됨 — 재학습 뒤 run_all 6a(eval --split val)를 다시 돌렸는지 확인")
        else:
            rec["target_threshold_json"] = None
        dst = os.path.join(RES_DIR, name)
        same = os.path.isfile(dst) and _hash(dst, "sha1") == sha1
        rec["resources_status"] = "동일(덮어쓰지 않음)" if same else ("교체" if os.path.isfile(dst) else "신규")
        manifest["teachers"].setdefault(mark, {})[kind] = rec
        print(f"  {name:42s} {size:>10,} B  sha1 {sha1[:10]}…  {rec['mtime'][:10]}  student 운영점 {rec.get('target_threshold')}  → {rec['resources_status']}")
    if problems and not args.allow_threshold_mismatch:
        print("[ERROR] 운영점 json 이 teacher 재학습보다 오래됐습니다(--allow_threshold_mismatch 로 강행 가능):"); [print("   -", p) for p in problems]
        sys.exit(1)
    if args.dry_run:
        print("[INFO] --dry_run: 복사·manifest 저장 안 함"); return

    os.makedirs(RES_DIR, exist_ok=True)
    targets = [RES_DIR] + [os.path.join(os.path.abspath(a), "resources") for a in args.also]
    for mark, kind, name, src, _ in plan:
        for tdir in targets:
            os.makedirs(tdir, exist_ok=True)
            dst = os.path.join(tdir, name)
            if os.path.isfile(dst) and _hash(dst, "sha1") == manifest["teachers"][mark][kind]["sha1"]:
                continue
            shutil.copyfile(src, dst)   # [수정 2026-09-21] drvfs(USB)에서 copy2 의 utime 이 PermissionError → copyfile
    for tdir in targets:
        with open(os.path.join(tdir, "teacher_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 16개 반입 완료 → {', '.join(targets)} (manifest: teacher_manifest.json)")


if __name__ == "__main__":
    main()
