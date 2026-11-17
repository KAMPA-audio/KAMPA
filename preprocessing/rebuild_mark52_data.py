#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_mark52_data.py — Mark5.2 · 전문가 8개(Mark4.1~4.8) · 평가 전용 세트를 한 번에 다시 만든다.

[신설 2026-09-15] 체크리스트 3절 "공용 재구성 스크립트". 09-07·09-11·09-15 결정을 코드로 옮긴 것.

■ 무엇이 바뀌는가 (09-07 실측: mark5.0 others 316 중 2/3 가 말소리·층간소음 경계 소리였다)
  talking  = TV 소리 + 사람 말소리, 50:50.  (옛 media_talking 은 TV 만)
             TV     : SINS watching_tv + CHiME v(말소리·타격음 없는 것) — 두 집 절반씩
             말소리 : CHiME c·f·m(v 없음, p 없음) + SINS social_activity — 두 집 절반씩
             ⚠ 말소리+타격음(B2)·TV+말소리 겹침(C) 은 쓰지 않는다(09-15 확정).
  others   = 층간소음이 아닌 먼 소리.  집 녹음 : 바깥 소리 = others_house_ratio(기본 0.5)
             집 녹음 : SINS cooking·working·eating + CHiME b·o(v·말소리·p 없음)
             바깥    : ESC-50 far_train 19종(can_opening·mouse_click 제외) + AI Hub 교통 4종·확성기
             ⚠ 옛 others 의 문·런닝머신·골프·악기·고양이·학원·골프연습장·에어컨·설거지·진공청소기는
               학습에서 빼고, 아래 평가 전용 세트로 보낸다(진공청소기는 그냥 제외).
  나머지 7종 = mark5.0 과 같은 파일을 그대로 복사(AI Hub). Mark4.8 만 6~29초 긴 파일을 3초로 정렬.

■ 만드는 것
  (A) Mark5.2   data_source_labeled/{9}  train 2000 / data_source_val 430 / data_source_test 430 /
                data_source_unlabeled 2000 (라벨 없이 평평, labeled 와 원본 안 겹침)
  (B) 전문가 8개 data/{train,val,test}/others_*  500 / 215 / 215  (8개가 같은 파일을 쓴다)
      Mark4.5   data/{split}/talking_*  500 / 215 / 215  (media_talking 대체)
      Mark4.8   dog_bark 긴 파일 → RMS 최대 3초 창으로 정렬(원본은 data/_source_long_20260915/)
      ⚠ 증강본(_aug_)은 여기서 안 만든다. 뒤에 augment_density.py --reset 을 전문가마다 돌린다.
  (C) Mark5.2   data_eval/zeroshot6 · boundary_B · far_heldout  (학습 금지, 평가 전용)
  (D) data_provenance.xlsx  mark5.2 행 신설, 전문가 옛 others·4.5 타겟·4.8 긴 파일 행은
      removed_20260715 열에 "replaced_20260915" 표시(파일은 _replaced_20260915/ 로 이동, 지우지 않음).

■ split 규칙 (원본이 정한다 — 같은 원본은 어디서나 같은 split)
  SINS·CHiME 세션 배정은 08-09 매니페스트 그대로. ESC-50 fold1-3/4/5. AI Hub 는 파일명 해시 70/15/15.
  [09-21] val·test 세션의 CHiME 'b'(광대역 생활소음) 청크는 어느 split 에도 쓰지 않는다(test 세션 한 곳에 34/35 가 몰려 있어
         세션 분할로는 train·val 에 줄 수 없고, train 으로 옮기면 세션이 겹친다). val·test 의 집 녹음 others 는 SINS 로만 채운다.
         (한 세션에 몰린 출처가 val 없이 test 로만 가는 것을 막는다. 검사: audit_dataset.py ⑧)
  Mark5.2 val·test 에 쓴 원본은 전문가 val·test 에서 뺀다. Mark5.2 labeled 와 unlabeled 는 안 겹친다.
  전문가 train 은 Mark5.2 train 과 겹쳐도 된다(09-07 방침). 단 안 쓴 원본을 먼저 쓴다.

■ 사용
  python preprocessing/rebuild_mark52_data.py --dry_run          # 풀 크기·배정 계획만 (게이트 결과는 캐시)
  python preprocessing/rebuild_mark52_data.py --stage all        # 실제 생성 (승인 뒤)
  --stage 는 m52 / experts / evalsets / provenance 를 쉼표로 고를 수 있다.
"""
import os, re, sys, csv, json, time, shutil, hashlib, argparse
from datetime import date
from collections import defaultdict, Counter

import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))            # KAMPA(옛 Mark5.2)
CODES_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))          # codes
MARK5_ROOT = os.path.join(CODES_ROOT, "Mark5")
EXPERT_ROOT = os.path.join(CODES_ROOT, "Mark4_series")
PROV_PATH = os.path.join(CODES_ROOT, "data_provenance.xlsx")
CACHE_DIR = os.path.join(BASE_DIR, "rebuild_cache")
# 원본 재료. 2026-09-27 WSL 홈(~/workspace)에서 USB 로 옮겼다. WSL 에서는
# wsl.exe -u root -- mount -t drvfs D: /mnt/d -o uid=1000,gid=1000,metadata 로 연결해야 보인다.
RAW_ROOT = "/mnt/d/workspace/journal/datasets_raw"
AIHUB_ROOT = os.path.join(RAW_ROOT, "aihub_raw/71296/extracted")
SINS_ROOT = os.path.join(RAW_ROOT, "sins_raw")
CHIME_ROOT = os.path.join(RAW_ROOT, "chime_home_raw/chime_home")
ESC_ROOT = os.path.join(RAW_ROOT, "esc50_clips")
SINS_MANIFEST = os.path.join(EXPERT_ROOT, "Mark4.5", "preprocessing", "sins_manifest.csv")

# mark5_slicer 의 창 선택·리샘플·스캔·게이트를 그대로 쓴다(전처리 규칙이 mark5.0 과 같아야 한다).
# [2026-11-18] Mark5_EXP2 를 USB 에서 지워, 같은 내용(sha256 일치)의 mark5_slicer.py·validate_dataset.py 가 있는 이 폴더에서 읽는다.
sys.path.insert(0, BASE_DIR)
from validate_dataset import check_waveform                                       # noqa: E402
from mark5_slicer import (scan_aihub, belongs, slice_resample, pick_window_aihub,  # noqa: E402
                          read_window_sins, read_window_chime, split_quota,
                          SINS_WINDOW_STARTS, CHIME_WINDOW_STARTS, SEG_SEC, TARGET_SR)
sys.path.insert(0, os.path.join(EXPERT_ROOT, "Mark4.5", "preprocessing"))
from chime_manifest import SESSION_SPLIT as CHIME_SESSION_SPLIT, LIST_FILES as CHIME_LIST_FILES, read_chunk_csv  # noqa: E402

MARK_VERSION = "mark5.2"
TODAY = "20260915"
REPLACED_TAG = f"replaced_{TODAY}"
CLASSES = ["heavy_impact", "dragging", "construction", "machine_noise",
           "talking", "water_toilet", "water_shower", "dog_bark", "others"]
OLD_CLASS = {"talking": "media_talking"}          # mark5.0 에서 복사할 때 이름 대응
COPY_CLASSES = [c for c in CLASSES if c not in ("talking", "others")]
SPLIT_TOTALS = {"train": 2000, "val": 430, "test": 430}
UNLABELED_TOTAL = 2000
SPLIT_DIRS = {"train": "data_source_labeled", "val": "data_source_val", "test": "data_source_test"}
EXPERT_OTHERS = {"train": 500, "val": 215, "test": 215}
EXPERT_TARGET = {"train": 500, "val": 215, "test": 215}   # Mark4.5 talking
EXPERT_VERSIONS = [f"mark4.{i}" for i in range(1, 9)]
EXPERT_TARGET_CLASS = {"mark4.1": "heavy_impact", "mark4.2": "dragging", "mark4.3": "construction",
                       "mark4.4": "machine_noise", "mark4.5": "talking", "mark4.6": "water_toilet",
                       "mark4.7": "water_shower", "mark4.8": "dog_bark"}

ESC_FAR_EXCLUDE = {"can_opening", "mouse_click"}
AIHUB_FAR = ["C.사업장_1.상가건물_a", "D.교통소음"]                 # 확성기 + 교통 4종
AIHUB_ZEROSHOT6 = ["2.경량충격음_b", "2.경량충격음_c", "2.경량충격음_d", "4.악기_a", "4.악기_b", "5.애완동물_b"]
AIHUB_BOUNDARY_B = ["C.사업장_1.상가건물_b", "C.사업장_2.학원시설", "C.사업장_3.골프연습장"]
SINS_HOUSE_OTHERS = {"cooking", "working", "eating"}
SINS_BOUNDARY_B = {"dishwashing"}
SPEECH, HOUSE = set("cfm"), set("pbo")


# ─────────────────────────────────────────────────────────────────────────────
# 풀 만들기.  item = dict(src=고유키, path, source, category, sess, split, group, loader)
#   group : 배정 때 균등하게 나눌 단위(카테고리 묶음).  loader : 3초 파형을 얻는 방법.
# ─────────────────────────────────────────────────────────────────────────────
def stable_split(name):
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def chime_label_set(meta, tier):
    if tier == "refined":
        labs = set(meta.get("majorityvote", "") or "") - set("SU")
        return labs, ("v" in labs)
    ann = [set(meta.get("annotation_a%d" % i, "") or "") for i in (1, 2, 3)]
    if not all(a and a not in ({"S"}, {"U"}) for a in ann):
        return set(), False
    vote = Counter(l for a in ann for l in a)
    maj = {l for l, c in vote.items() if c >= 2 and l not in "SU"}
    n_v = sum(1 for a in ann if "v" in a)
    if 0 < n_v < 3:
        return set(), False
    return maj, (n_v == 3)


def chime_role(labs, v_ok):
    """tv / speech / house / None.  09-15 규칙: p(타격음) 든 것은 어디에도 안 쓴다."""
    if not labs or "p" in labs:
        return None
    sp, ho = bool(labs & SPEECH), bool(labs & HOUSE)
    if v_ok:
        return None if sp else "tv"            # v 만(+b·o 허용). v+말소리(C) 는 제외
    if sp:
        return "speech"                        # A + B1
    if ho:
        return "house"                         # E1 (b·o)
    return None


def build_chime():
    chunk = os.path.join(CHIME_ROOT, "chunks")
    seen, out = set(), []
    for tier, lf in CHIME_LIST_FILES:
        for line in open(os.path.join(CHIME_ROOT, lf), encoding="utf-8"):
            line = line.strip()
            if not line or "," not in line:
                continue
            name = line.split(",", 1)[1]
            if name in seen:
                continue
            seen.add(name)
            sess = name.split(".s")[0]
            split = CHIME_SESSION_SPLIT.get(sess)
            if split is None:
                continue
            labs, v_ok = chime_label_set(read_chunk_csv(os.path.join(chunk, name + ".csv")), tier)
            role = chime_role(labs, v_ok)
            if role is None:
                continue
            wav = name + ".16kHz.wav"
            if not os.path.exists(os.path.join(chunk, wav)):
                continue
            # [수정 2026-09-21] 광대역 생활소음 'b' 청크는 35개 중 34개가 test 세션 230110_1036 한 곳에 몰려 있어 세션 단위 분할이
            # 그 덩어리를 통째로 test 에 보냈다(전문가 others train 0 / val 0 / test 29). construction 전문가가 val 에서 못 본 소리를
            # test 에서 construction 으로 판정(Others FPR 0.9%→13.5%). talking 도 b 든 말소리 청크 21개 중 19개가 test 세션이었다.
            # 처음엔 "b 는 train 전용" 으로 옮겼으나 그러면 train 과 test 가 같은 세션을 공유해 audit ② 세션 겹침이 깨진다(03:00 실측).
            # 최종 규칙: val·test 세션의 b 청크는 어느 split 에도 쓰지 않는다. train 세션의 b·o 는 그대로 train. 검사: audit ⑧.
            if "b" in labs and split != "train":
                continue
            out.append(dict(src=wav, path=os.path.join(chunk, wav), source="chime_home",
                            category="".join(sorted(labs)), sess=sess.replace("CR_lounge_", ""),
                            split=split, role=role, group=f"chime_{role}", loader="chime",
                            volume=sess.replace("CR_lounge_", "")))
    return out


def build_sins():
    out = []
    for r in csv.DictReader(open(SINS_MANIFEST, encoding="utf-8")):
        act = r["activity"]
        if act == "watching_tv":
            role = "tv"
        elif act == "social_activity":
            role = "speech"
        elif act in SINS_HOUSE_OTHERS:
            role = "house"
        elif act in SINS_BOUNDARY_B:
            role = "boundary_B"
        else:
            continue                                   # vacuum_cleaner 는 제외(09-11 결정)
        path = os.path.join(SINS_ROOT, r["split"], r["role"], r["wav"])
        if not os.path.exists(path):
            continue
        # [수정 2026-09-15] 평가 전용(boundary_B: dishwashing)은 split 을 "eval" 로 둔다. 매니페스트의
        # train/val/test 를 그대로 두면 plan_all 이 (role,"eval") 풀에서 찾지 못해 143개가 빠졌다(1차 생성).
        out.append(dict(src=r["wav"], path=path, source="sins", category=act, sess=r["sess"],
                        split=("eval" if role == "boundary_B" else r["split"]), role=role,
                        group=("sins_tv" if role == "tv" else "sins_social" if role == "speech"
                               else f"sins_{act}"), loader="sins", volume=f"DevNode{r['node']}"))
    return out


def build_esc():
    out = []
    for r in csv.DictReader(open(os.path.join(ESC_ROOT, "esc50_manifest.csv"), encoding="utf-8-sig")):
        cat, fold, role = r["aihub_category"], r["aihub_volume"], r["role"]
        path = os.path.join(ESC_ROOT, r["rel_path"])
        if not os.path.exists(path):
            continue
        if role == "far_train":
            if cat in ESC_FAR_EXCLUDE:
                continue
            split = "train" if fold in ("fold1", "fold2", "fold3") else ("val" if fold == "fold4" else "test")
            rr = "far"
        elif role == "far_heldout":
            split, rr = "eval", "far_heldout"
        else:
            split, rr = "eval", "boundary_B"
        out.append(dict(src=r["aihub_src_file"], path=path, source="esc50", category=cat, sess=fold,
                        split=split, role=rr, group=f"esc_{cat}", loader="copy", volume=fold,
                        esc_row=r))
    return out


def build_aihub():
    out = []
    for c in scan_aihub(AIHUB_ROOT):
        cat = c["category"]
        if belongs(cat, AIHUB_FAR):
            role = "far"
        elif belongs(cat, AIHUB_ZEROSHOT6):
            role = "zeroshot6"
        elif belongs(cat, AIHUB_BOUNDARY_B):
            role = "boundary_B"
        else:
            continue
        for w in c["wavs"]:
            lp = os.path.join(c["label_dir"], os.path.splitext(w)[0] + ".json") if c["label_dir"] else None
            out.append(dict(src=w, path=os.path.join(c["dir"], w), label=lp, source="aihub",
                            category=cat, sess="", split=(stable_split(w) if role == "far" else "eval"),
                            role=role, group=f"aihub_{cat.split('_')[-1][:12]}", loader="aihub",
                            volume=c["volume"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 게이트 + 창 선택 (결과는 캐시 — dry run 을 여러 번 돌려도 오디오를 다시 읽지 않는다)
# ─────────────────────────────────────────────────────────────────────────────
def load_segment(item):
    """(seg, start) 를 돌려준다. 게이트 탈락이면 (None, reason)."""
    if item["loader"] == "copy":
        x, sr = sf.read(item["path"], dtype="float32", always_2d=True)
        seg = np.ascontiguousarray(x[:, 0])
        r = item["esc_row"]
        return seg, float(r.get("seg_start_sec") or 0.0)
    if item["loader"] == "aihub":
        x, sr, start, end, _ = pick_window_aihub(item["path"], item.get("label"))
        seg = slice_resample(x, sr, start, end)
        bad = check_waveform(seg, TARGET_SR, min_duration=SEG_SEC)
        return (seg, start) if bad is None else (None, bad)
    starts = SINS_WINDOW_STARTS if item["loader"] == "sins" else CHIME_WINDOW_STARTS
    reader = read_window_sins if item["loader"] == "sins" else read_window_chime
    last = "?"
    for st in starts:
        try:
            seg = reader(item["path"], st, remove_dc=True)
        except Exception as e:
            last = f"읽기실패 {e}"
            continue
        bad = check_waveform(seg, TARGET_SR, min_duration=SEG_SEC)
        if bad is None:
            return seg, st
        last = bad
    return None, last


def gate_pool(items, cache):
    """캐시에 없는 것만 오디오를 읽어 게이트를 건다. cache[src] = {"pass":bool,"start":float,"why":str}"""
    todo = [it for it in items if it["src"] not in cache]
    t0 = time.time()
    for i, it in enumerate(todo, 1):
        if it["loader"] == "copy":
            cache[it["src"]] = {"pass": True, "start": float(it["esc_row"].get("seg_start_sec") or 0.0), "why": ""}
            continue
        seg, info = load_segment(it)
        cache[it["src"]] = ({"pass": True, "start": float(info), "why": ""} if seg is not None
                            else {"pass": False, "start": None, "why": str(info)})
        if i % 200 == 0:
            print(f"    게이트 {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    return [it for it in items if cache[it["src"]]["pass"]]


# ─────────────────────────────────────────────────────────────────────────────
# 배정.  draw(items, want, used) — group 균등 + group 안에서 sess 라운드로빈.
#   used 에 든 src 는 건너뛴다(사용 뒤 used 에 넣는 것은 호출자).  prefer_fresh 가 주어지면
#   그 집합에 없는 것을 먼저 쓴다(전문가 train 이 Mark5.2 train 과 겹치는 것을 최소화).
# ─────────────────────────────────────────────────────────────────────────────
def draw(items, want, used, prefer_fresh=None):
    pool = [it for it in items if it["src"] not in used]
    if prefer_fresh is not None:
        pool = [it for it in pool if it["src"] not in prefer_fresh] + \
               [it for it in pool if it["src"] in prefer_fresh]
    by_g = defaultdict(list)
    for it in pool:
        by_g[it["group"]].append(it)
    groups = sorted(by_g)
    n = len(groups)
    if n == 0 or want <= 0:
        return []
    base, rest = divmod(want, n)
    quota = {g: base + (1 if i < rest else 0) for i, g in enumerate(groups)}
    picked, leftover = [], []
    for g in groups:
        by_s = defaultdict(list)
        for it in by_g[g]:
            by_s[it["sess"]].append(it)
        got = []
        while len(got) < quota[g] and any(by_s.values()):
            for s in sorted(by_s):
                if by_s[s] and len(got) < quota[g]:
                    got.append(by_s[s].pop(0))
        picked.extend(got)
        leftover.append([it for s in sorted(by_s) for it in by_s[s]])
    # 모자란 만큼 예비에서 group 을 번갈아 채운다
    while len(picked) < want and any(leftover):
        for lst in leftover:
            if lst and len(picked) < want:
                picked.append(lst.pop(0))
    return picked[:want]


def plan_all(pools, house_ratio):
    """전체 배정을 계산한다(파일은 안 만든다). 결과: dict[(dataset, split, cls)] -> [items]"""
    P = defaultdict(list)          # (source-role, split) -> items
    for it in pools:
        P[(it["role"], it["split"])].append(it)
    used_m52 = {"train": set(), "val": set(), "test": set()}   # Mark5.2 가 쓴 원본(split 별)
    plan, short = {}, []

    notes = []

    def take(key, role, split, want, used, prefer_fresh=None, strict=True):
        got = draw(P[(role, split)], want, used, prefer_fresh)
        plan.setdefault(key, []).extend(got)
        for it in got:
            used.add(it["src"])
        if len(got) < want:
            (short if strict else notes).append((key, role, want, len(got)))
        return got

    def talking(key, split, want, used, prefer_fresh=None):
        n_tv = want // 2
        take(key, "tv", split, n_tv, used, prefer_fresh)
        take(key, "speech", split, want - n_tv, used, prefer_fresh)

    def others(key, split, want, used, prefer_fresh=None):
        # 집 녹음이 모자라면 그 몫은 바깥 소리로 채운다(비율은 어림값이라 몇 개 차이는 허용).
        # talking 의 TV:말소리 50:50 은 val·test 반반이 규칙이라 이런 보정을 두지 않는다.
        n_house = int(round(want * house_ratio))
        got_h = take(key, "house", split, n_house, used, prefer_fresh, strict=False)
        take(key, "far", split, want - len(got_h), used, prefer_fresh)

    # (A) Mark5.2 labeled  — val → test → train
    q = {sp: split_quota(SPLIT_TOTALS[sp], CLASSES) for sp in SPLIT_TOTALS}
    for sp in ("val", "test", "train"):
        talking(("m52", sp, "talking"), sp, q[sp]["talking"], used_m52[sp])
        others(("m52", sp, "others"), sp, q[sp]["others"], used_m52[sp])
    # (A') unlabeled — train 풀에서, labeled 와 안 겹치게 (used_m52["train"] 을 그대로 씀)
    uq = split_quota(UNLABELED_TOTAL, CLASSES)
    talking(("m52", "unlabeled", "talking"), "train", uq["talking"], used_m52["train"])
    others(("m52", "unlabeled", "others"), "train", uq["others"], used_m52["train"])
    m52_train_used = set(used_m52["train"])
    # (B) 전문가 others (8개 공통) — val·test 는 Mark5.2 val·test 원본 제외, train 은 안 쓴 것 우선
    for sp in ("val", "test"):
        others(("expert", sp, "others"), sp, EXPERT_OTHERS[sp], set(used_m52[sp]))
    others(("expert", "train", "others"), "train", EXPERT_OTHERS["train"], set(), prefer_fresh=m52_train_used)
    # (B') Mark4.5 talking 타겟
    for sp in ("val", "test"):
        talking(("expert", sp, "talking"), sp, EXPERT_TARGET[sp], set(used_m52[sp]))
    talking(("expert", "train", "talking"), "train", EXPERT_TARGET["train"], set(), prefer_fresh=m52_train_used)
    # (C) 평가 전용 — 전부
    for role in ("zeroshot6", "boundary_B", "far_heldout"):
        plan[("eval", role, role)] = list(P[(role, "eval")])
    return plan, short, P, notes


def print_plan(plan, short, P, cache, notes=()):
    print("\n[풀 크기 — 게이트 통과 / 후보]")
    for (role, sp) in sorted(P):
        items = P[(role, sp)]
        by = Counter(it["group"] for it in items)
        print(f"  {role:11s} {sp:6s} {len(items):5d}   " + ", ".join(f"{g} {n}" for g, n in sorted(by.items())))
    print("\n[배정 계획]")
    for key in sorted(plan, key=lambda k: (k[0], {"val": 0, "test": 1, "train": 2, "unlabeled": 3}.get(k[1], 9), k[2])):
        items = plan[key]
        by = Counter(it["group"] for it in items)
        print(f"  {key[0]:6s} {key[1]:9s} {key[2]:10s} {len(items):5d}   " + ", ".join(f"{g} {n}" for g, n in sorted(by.items())))
    if short:
        print("\n[부족]")
        for key, role, want, got in short:
            print(f"  {key} {role}: 필요 {want}, 채움 {got}")
    else:
        print("\n[부족] 없음")
    for key, role, want, got in notes:
        print(f"[비고] {key} {role} {want} 중 {got} 만 있어 나머지 {want-got} 은 바깥 소리로 채움")


# ─────────────────────────────────────────────────────────────────────────────
# 쓰기
# ─────────────────────────────────────────────────────────────────────────────
def prov_row(name, cls, split, item, seg_start, path, version=MARK_VERSION, extra=None):
    row = {"local_filename": name, "original_labels": item["category"], "target_class": cls,
           "assigned_split": split, "mark_version": version,
           "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
           "size_bytes": os.path.getsize(path), "download_date": date.today().isoformat(),
           "source_type": "original", "removed_20260715": "active", "source": item["source"],
           "aihub_src_file": item["src"], "aihub_category": item["category"], "aihub_volume": item["volume"],
           "seg_start_sec": round(float(seg_start), 3), "seg_end_sec": round(float(seg_start) + SEG_SEC, 3)}
    if extra:
        row.update(extra)
    return row


def write_items(items, out_dir, name_fn, cls, split, cache, rows, version=MARK_VERSION, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    for i, it in enumerate(items, 1):
        name = name_fn(i)
        path = os.path.join(out_dir, name)
        if it["loader"] == "copy":
            shutil.copy2(it["path"], path)
            start = cache[it["src"]]["start"]
        else:
            seg, start = load_segment(it)
            if seg is None:
                raise SystemExit(f"[ERROR] 게이트 통과 캐시와 다름: {it['src']} {start}")
            sf.write(path, seg, TARGET_SR, subtype="FLOAT")
        rows.append(prov_row(name, cls, split, it, start, path, version, extra))


def rms_best_3s(path):
    """긴 파일에서 RMS 가 가장 큰 3초 창(0.1초 간격). mark5_slicer/esc50_slicer 규칙과 같다."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    n = int(SEG_SEC * sr)
    if len(x) <= n:
        return x, 0.0
    hop = int(0.1 * sr)
    best, best_s = -1.0, 0
    for s in range(0, len(x) - n + 1, hop):
        r = float(np.sqrt(np.mean(x[s:s + n] ** 2)))
        if r > best:
            best, best_s = r, s
    return x[best_s:best_s + n], best_s / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stage", type=str, default="all", help="all 또는 m52,experts,evalsets,provenance")
    ap.add_argument("--others_house_ratio", type=float, default=0.5)
    ap.add_argument("--provenance_path", type=str, default=PROV_PATH)
    args = ap.parse_args()
    stages = {"m52", "experts", "evalsets", "provenance"} if args.stage == "all" else set(args.stage.split(","))
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "gate_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    t0 = time.time()
    print(f"[INFO] mark_version={MARK_VERSION}  클래스 {CLASSES}")
    print(f"[INFO] others 집 녹음 비율 {args.others_house_ratio}")
    pools_raw = build_chime() + build_sins() + build_esc() + build_aihub()
    print(f"[INFO] 후보 {len(pools_raw)}개 — 게이트 검사(캐시 {len(cache)}개)")
    pools = gate_pool(pools_raw, cache)
    json.dump(cache, open(cache_path, "w"))
    fail = Counter((it["source"], it["role"]) for it in pools_raw if not cache[it["src"]]["pass"])
    print(f"[INFO] 게이트 통과 {len(pools)} / 탈락 {sum(fail.values())}  " +
          ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(fail.items())) + f"  ({time.time()-t0:.0f}s)")

    plan, short, P, notes = plan_all(pools, args.others_house_ratio)
    print_plan(plan, short, P, cache, notes)
    json.dump({"|".join(k): [it["src"] for it in v] for k, v in plan.items()},
              open(os.path.join(CACHE_DIR, "plan.json"), "w"), ensure_ascii=False, indent=0)
    if short:
        print("\n[ERROR] 부족한 칸이 있어 중단합니다.")
        sys.exit(1)
    if args.dry_run:
        print("\n[DRY RUN] 파일은 만들지 않았습니다. 계획은 rebuild_cache/plan.json")
        return

    # ── provenance 읽기(한 번) ──
    import pandas as pd
    prov = pd.read_excel(args.provenance_path)
    backup = args.provenance_path + f".bak_before_mark52_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(args.provenance_path, backup)
    print(f"\n[INFO] provenance {len(prov)}행 읽음, 백업 {os.path.basename(backup)}")
    new_rows = []
    m50 = prov[prov["mark_version"] == "mark5.0"]

    # ── (A) Mark5.2 ──
    if "m52" in stages:
        print("\n##### (A) Mark5.2 #####", flush=True)
        # 7종 복사 — labeled/val/test
        for sp in ("val", "test", "train"):
            for cls in COPY_CLASSES:
                src_dir = os.path.join(MARK5_ROOT, SPLIT_DIRS[sp], cls)
                dst_dir = os.path.join(PROJECT_ROOT, SPLIT_DIRS[sp], cls)
                os.makedirs(dst_dir, exist_ok=True)
                files = sorted(f for f in os.listdir(src_dir) if f.endswith(".wav"))
                for f in files:
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                    old = m50[(m50["local_filename"] == f) & (m50["assigned_split"] == sp)]
                    if len(old) != 1:
                        raise SystemExit(f"[ERROR] mark5.0 provenance 에 {sp}/{f} 행이 {len(old)}개")
                    r = old.iloc[0].to_dict()
                    r.update({"mark_version": MARK_VERSION, "download_date": date.today().isoformat()})
                    new_rows.append(r)
                print(f"  {sp:5s} {cls:14s} 복사 {len(files)}", flush=True)
            # talking / others 신규
            for cls in ("talking", "others"):
                items = plan[("m52", sp, cls)]
                write_items(items, os.path.join(PROJECT_ROOT, SPLIT_DIRS[sp], cls),
                            lambda i, c=cls, s=sp: f"{c}_{s}_{i:03d}.wav", cls, sp, cache, new_rows)
                print(f"  {sp:5s} {cls:14s} 생성 {len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        # unlabeled — 7종은 mark5.0 unlabeled 에서 복사(번호 새로), talking/others 는 신규
        udir = os.path.join(PROJECT_ROOT, "data_source_unlabeled")
        os.makedirs(udir, exist_ok=True)
        k = 0
        mu = m50[m50["assigned_split"] == "unlabeled"]
        for cls in COPY_CLASSES:
            sub = mu[mu["target_class"] == f"unlabeled:{cls}"].sort_values("local_filename")
            for _, r in sub.iterrows():
                k += 1
                name = f"unlabeled_{k:04d}.wav"
                shutil.copy2(os.path.join(MARK5_ROOT, "data_source_unlabeled", r["local_filename"]),
                             os.path.join(udir, name))
                rr = r.to_dict(); rr.update({"local_filename": name, "mark_version": MARK_VERSION,
                                             "download_date": date.today().isoformat()})
                new_rows.append(rr)
            print(f"  unlabeled {cls:14s} 복사 {len(sub)}", flush=True)
        for cls in ("talking", "others"):
            items = plan[("m52", "unlabeled", cls)]
            start_k = k
            write_items(items, udir, lambda i, s=start_k: f"unlabeled_{s + i:04d}.wav",
                        f"unlabeled:{cls}", "unlabeled", cache, new_rows)
            k += len(items)
            print(f"  unlabeled {cls:14s} 생성 {len(items)}  (누적 {k})", flush=True)

    # ── (B) 전문가 ──
    if "experts" in stages:
        print("\n##### (B) 전문가 8개 #####", flush=True)
        stage_dir = os.path.join(CACHE_DIR, "expert_common")
        common_rows = {}                      # (split) -> rows(템플릿, mark_version 은 나중에)
        for sp in ("val", "test", "train"):
            items = plan[("expert", sp, "others")]
            rows = []
            write_items(items, os.path.join(stage_dir, sp), lambda i, s=sp: f"others_{s}_{i:03d}.wav",
                        "others", sp, cache, rows, version="__EXPERT__")
            common_rows[sp] = rows
            print(f"  공통 others {sp:5s} {len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        talk_rows = {}
        for sp in ("val", "test", "train"):
            items = plan[("expert", sp, "talking")]
            rows = []
            write_items(items, os.path.join(stage_dir, "talking", sp), lambda i, s=sp: f"talking_{s}_{i:03d}.wav",
                        "talking", sp, cache, rows, version="mark4.5")
            talk_rows[sp] = rows
            print(f"  4.5 talking {sp:5s} {len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        for ver in EXPERT_VERSIONS:
            root = os.path.join(EXPERT_ROOT, "Mark" + ver[4:])
            data = os.path.join(root, "data")
            rep = os.path.join(data, f"_replaced_{TODAY}")
            moved = 0
            for sp in ("train", "val", "test"):
                d = os.path.join(data, sp)
                os.makedirs(os.path.join(rep, sp), exist_ok=True)
                for f in os.listdir(d):
                    if f.startswith("others_") or (ver == "mark4.5" and f.startswith("media_talking_")) \
                            or "_aug_" in f:
                        shutil.move(os.path.join(d, f), os.path.join(rep, sp, f)); moved += 1
                for f in os.listdir(os.path.join(stage_dir, sp)):
                    shutil.copy2(os.path.join(stage_dir, sp, f), os.path.join(d, f))
                for r in common_rows[sp]:
                    rr = dict(r); rr["mark_version"] = ver; new_rows.append(rr)
                if ver == "mark4.5":
                    for f in os.listdir(os.path.join(stage_dir, "talking", sp)):
                        shutil.copy2(os.path.join(stage_dir, "talking", sp, f), os.path.join(d, f))
                    new_rows.extend(dict(r) for r in talk_rows[sp])
            # provenance 옛 행 표시: others 전부 + 증강본 전부 (+4.5 타겟)
            m = (prov["mark_version"] == ver) & (prov["removed_20260715"] == "active") & (
                (prov["target_class"] == "others") | (prov["source_type"] != "original")
                | ((ver == "mark4.5") & (prov["target_class"] == "media_talking")))
            prov.loc[m, "removed_20260715"] = REPLACED_TAG
            print(f"  {ver}: 옛 파일 {moved}개 → _replaced_{TODAY}/, provenance 표시 {int(m.sum())}행", flush=True)
            if ver == "mark4.8":
                # dog_bark 긴 파일 정렬
                src_dir = os.path.join(data, f"_source_long_{TODAY}")
                n_long = 0
                for sp in ("train", "val", "test"):
                    d = os.path.join(data, sp)
                    for f in sorted(os.listdir(d)):
                        if not (f.startswith("dog_bark_") and f.endswith(".wav")):
                            continue
                        p = os.path.join(d, f)
                        info = sf.info(p)
                        if info.frames <= int(SEG_SEC * info.samplerate) + 1:
                            continue
                        os.makedirs(os.path.join(src_dir, sp), exist_ok=True)
                        shutil.move(p, os.path.join(src_dir, sp, f))
                        seg, st = rms_best_3s(os.path.join(src_dir, sp, f))
                        sf.write(p, seg.astype(np.float32), info.samplerate, subtype="FLOAT")
                        old = prov[(prov["mark_version"] == ver) & (prov["local_filename"] == f)
                                   & (prov["removed_20260715"] == "active")]
                        if len(old) == 1:
                            r = old.iloc[0].to_dict()
                            prov.loc[old.index, "removed_20260715"] = REPLACED_TAG
                            r.update({"removed_20260715": "active", "seg_start_sec": round(st, 3),
                                      "seg_end_sec": round(st + SEG_SEC, 3), "duration_sec": SEG_SEC,
                                      "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                                      "size_bytes": os.path.getsize(p), "mark52_note": f"realign3s_{TODAY}"})
                            new_rows.append(r)
                        n_long += 1
                print(f"  mark4.8: dog_bark 긴 파일 {n_long}개 → 3초 정렬(원본 _source_long_{TODAY}/)", flush=True)

    # ── (C) 평가 전용 ──
    if "evalsets" in stages:
        print("\n##### (C) 평가 전용 세트 #####", flush=True)
        for role in ("zeroshot6", "boundary_B", "far_heldout"):
            items = plan[("eval", role, role)]
            by_cat = defaultdict(list)
            for it in items:
                by_cat[it["category"]].append(it)
            for cat, its in sorted(by_cat.items()):
                short_cat = re.sub(r"[^0-9A-Za-z가-힣]+", "_", cat)[-40:]
                write_items(its, os.path.join(PROJECT_ROOT, "data_eval", role, short_cat),
                            lambda i, r=role, c=short_cat: f"{r}_{c}_{i:03d}.wav",
                            f"eval:{role}", f"eval_{role}", cache, new_rows)
            print(f"  {role:12s} {len(items)}개 / {len(by_cat)}카테고리  ({time.time()-t0:.0f}s)", flush=True)

    # ── (D) provenance 쓰기 ──
    if "provenance" in stages or new_rows:
        import pandas as pd
        # SINS 진공청소기(옛 mark5.0 others) 제외 표시 — 파일은 그대로
        m = (prov["mark_version"] == "mark5.0") & (prov["source"] == "sins") & (prov["aihub_category"] == "vacuum_cleaner")
        if "mark52_note" not in prov.columns:
            prov["mark52_note"] = ""
        prov.loc[m, "mark52_note"] = f"excluded_{TODAY}(vacuum_cleaner)"
        merged = pd.concat([prov, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_excel(args.provenance_path, index=False)
        print(f"\n[provenance] 신규 {len(new_rows)}행 추가 → 총 {len(merged)}행, 진공청소기 표시 {int(m.sum())}행  ({time.time()-t0:.0f}s)")
    print(f"\n[완료] {(time.time()-t0)/60:.1f}분. 다음: validate_dataset → audit_dataset, 전문가는 augment_density --reset")


if __name__ == "__main__":
    main()
