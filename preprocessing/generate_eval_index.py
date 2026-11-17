# -*- coding: utf-8 -*-
"""평가 전용 세트(data_eval/) 인덱스 3종 생성 — 체크리스트 10절 "평가 세트 인덱스 3종(경계 6종, 경계 B, ESC-50 far_heldout)".

[신설 2026-09-17 / Mark5.2]

data_eval/ 은 2026-09-15 재구성 때 만든 **학습 금지** 평가 세트다(rebuild_mark52_data.py (C) 단계).
    zeroshot6/<AI Hub 폴더>/*.wav    제로샷 6종 — 문여닫기·런닝머신·골프 퍼팅·바이올린·피아노·고양이 (AI Hub 원본 전체, 631)
    boundary_B/<카테고리>/*.wav       경계 B — AI Hub 학원·골프연습장·에어컨 실외기 + SINS 설거지 + ESC-50 near 19종 (1,272)
    far_heldout/<ESC-50 카테고리>/*.wav  ESC-50 far_heldout 10종 (381) — 학습 others 풀에 안 넣은 먼 소리

만드는 CSV (헤더: path,label,category):
    dataset_eval_zeroshot6.csv    label = 영어 클래스명 6종(아래 ZEROSHOT_CLASSES) — 제로샷 표(새 클래스 재현율·오분류 방향)
    dataset_eval_boundary_B.csv   label = others,  category = 폴더명 — open-set ① "학습에 없던 소리를 others 로 보낸 비율"
    dataset_eval_far_heldout.csv  label = others,  category = 폴더명 — open-set ① (먼 층)
  open-set 세트는 정답이 전부 others 이므로 eval.py 의 Recall(others) = 1 − "Others FPR" 이 곧 ① 이다.
  eval.py --split eval_far_heldout 처럼 부르면 dataset_eval_far_heldout.csv 를 읽는다(2026-09-17 eval.py 확장).

경로는 다른 인덱스와 같이 이 저장소의 절대경로로 적고, 루트·preprocessing·model 세 곳에 같은 파일을 둔다(mirror).
[수정 2026-09-21] 폴더명은 NFC 로 정규화해 대조·기록한다 — 코랩 드라이브 마운트가 한글 폴더명을 NFD 로 돌려줘 [셀 16b] 가 멈췄던 것.

사용:  python preprocessing/generate_eval_index.py            # 3종 전부
       python preprocessing/generate_eval_index.py --set zeroshot6
"""
import argparse
import csv
import glob
import os
import sys
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data_eval")

# 제로샷 6종: AI Hub 폴더명 -> 영어 클래스명. 문장(프롬프트)은 shared_vild/resources/zeroshot_prompt_bank.json 에 있다.
ZEROSHOT_CLASSES = {
    "A_층간소음_2_경량충격음_b_문여닫는소리": "door",
    "A_층간소음_2_경량충격음_c_런닝머신에서뛰는소리": "treadmill",
    "A_층간소음_2_경량충격음_d_골프퍼팅_골굴리는소리_": "golf_putting",
    "A_층간소음_4_악기_a_바이올린연주소리": "violin",
    "A_층간소음_4_악기_b_피아노연주소리": "piano",
    "A_층간소음_5_애완동물_b_고양이우는소리": "cat",
}
SETS = {"zeroshot6": "zeroshot", "boundary_B": "others", "far_heldout": "others"}


def _mirror(entries, filename):
    header = ["path", "label", "category"]
    for target in (os.path.join(PROJECT_ROOT, filename), os.path.join(BASE_DIR, filename),
                   os.path.join(PROJECT_ROOT, "model", filename)):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(entries)
    print(f"[완료] {filename}: {len(entries)}행 (루트·preprocessing·model 세 곳)")


def build(set_name):
    root = os.path.join(EVAL_DIR, set_name)
    if not os.path.isdir(root):
        raise SystemExit(f"[ERROR] 평가 세트 폴더가 없습니다: {root} (data_eval 오디오가 이 저장소에 있어야 한다)")
    entries, counts = [], {}
    for cat_fs in sorted(os.listdir(root)):
        cdir = os.path.join(root, cat_fs)              # 경로는 파일시스템이 준 이름 그대로 (파일을 읽는 데 쓴다)
        if not os.path.isdir(cdir):
            continue
        # [수정 2026-09-21] 코랩의 드라이브 마운트가 한글 폴더명을 자모 분리(NFD)로 돌려줘 ZEROSHOT_CLASSES 대조가
        # "모르는 폴더" 로 실패했다(드라이브·로컬·코드 키는 전부 NFC). 대조와 CSV 의 category 열은 NFC 로 맞춘 이름을 쓴다.
        cat = unicodedata.normalize("NFC", cat_fs)
        files = sorted(glob.glob(os.path.join(cdir, "*.wav")))
        if SETS[set_name] == "zeroshot":
            if cat not in ZEROSHOT_CLASSES:
                raise SystemExit(f"[ERROR] zeroshot6 에 모르는 폴더: {cat} (ZEROSHOT_CLASSES 에 추가하십시오)")
            label = ZEROSHOT_CLASSES[cat]
        else:
            label = "others"
        for p in files:
            entries.append([os.path.abspath(p), label, cat])
        counts[label if SETS[set_name] == "zeroshot" else cat] = counts.get(label if SETS[set_name] == "zeroshot" else cat, 0) + len(files)
    if not entries:
        raise SystemExit(f"[ERROR] {set_name}: wav 가 하나도 없습니다.")
    _mirror(entries, f"dataset_eval_{set_name}.csv")
    for k, v in sorted(counts.items()):
        print(f"  - {k}: {v}")
    return len(entries)


def main():
    ap = argparse.ArgumentParser(description="data_eval 인덱스 생성")
    ap.add_argument("--set", type=str, default="all", choices=["all"] + list(SETS))
    args = ap.parse_args()
    names = list(SETS) if args.set == "all" else [args.set]
    total = 0
    for n in names:
        total += build(n)
    print(f"[INFO] 합계 {total}행 ({', '.join(names)})")


if __name__ == "__main__":
    main()
