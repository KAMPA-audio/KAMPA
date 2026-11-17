#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gate_external_candidates.py — 교사 train 부족분을 채울 외부 양성 후보에 기존 품질 게이트를 건다 (2026-10-23).

왜: 교사 train 이 Mark5.3 학습 쪽(정답 라벨·라벨 없는 클립)과 같은 녹음을 공유하지 않게 하려면, 교사 train 을 Mark5.3 이
    쓰지 않은 녹음으로만 채워야 한다. dragging·water_toilet·water_shower 는 그런 AI Hub 녹음이 393~408개뿐이라
    외부 자료로 500개를 채운다. 이 스크립트는 후보를 고르고 검사만 한다(교사 데이터·provenance 는 바꾸지 않는다).
무엇을:
    출처        클래스                       원본 위치 (EXT_ROOT 아래)            이용조건
    FSD50K      water_toilet·water_shower·   fsd50k_audio/<id>.wav               CC0 / CC BY 3.0 / CC BY-NC 3.0 / Sampling+
                dragging·dog_bark
    Clotho 2.1  water_shower                 clotho/audio/<split>/...            Freesound 원 라이선스(CSV)
    BeDAViN     water_toilet·water_shower    bedavin/<cls>__<id>.wav             CC BY 4.0
    BBC SFX     water_shower                 bbc_sfx/<id>.wav                    RemArc(연구·교육용)
    SNU-B36-50E dragging(의자 끌기 CD)        snu_b36_50e/cd_wav/<track>.wav      GPL-3.0
  - 녹음 하나에서 3초 한 개만 쓴다(원본 1개 = 녹음 1개). 창은 rms_best_3s 와 같은 규칙(0.1초 간격, 첫 채널)으로 RMS 순으로
    보되, 가장 큰 창이 게이트(check_waveform: 길이·rms·peak·클리핑)에서 떨어지면 RMS 가 최댓값의 절반 이상인 다음 창을 본다.
  - 이미 쓴 원본은 뺀다: provenance 전 행(삭제 행 포함)의 FSD50K ID, ESC-50 의 Freesound ID. Clotho 는 FSD50K 와 같은
    Freesound ID 가 있으면 FSD50K 쪽만 남긴다.
  - 내용 검사: AudioSet mn10 태거로 물소리·개 클래스의 3초 창이 실제 그 소리인지 본다(기준은 AI Hub 정답 라벨 원본의 5 백분위, 아래 상수 설명).
  - 같은 녹음이 다른 이름으로 들어온 경우(BBC 재수록판 등)를 잡으려고 클래스 안에서 log-mel 상관 ≥ 0.98 인 쌍을 표시한다.
산출물: EXT_ROOT/gated/<cls>/<출처>_<id>.wav (16kHz float, 3초) · EXT_ROOT/gated/gate_report[_<tag>].csv
사용: python preprocessing/gate_external_candidates.py [--ext_root /mnt/c/Users/user/journal_external_raw]
"""
import os, re, sys, csv, json, glob, random, hashlib, argparse
from collections import Counter

import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import rebuild_mark52_data as R                                    # noqa: E402
from validate_dataset import check_waveform                        # noqa: E402
from mark5_slicer import slice_resample, TARGET_SR, SEG_SEC        # noqa: E402

EXT_ROOT_DEFAULT = "/mnt/c/Users/user/journal_external_raw"
# [10-25] 재현 기록: 후보 목록·선택 결과의 사본을 모델 저장소(KAMPA, 구 Mark5.3 의 reproducibility/external_sources/)에 둔다. GPT 후보 목록은
#         원래 C드라이브의 GPT 작업 폴더(.codex)에서 읽었는데, 그 폴더가 없는 환경에서도 같은 후보로 다시 돌릴 수 있게 그 사본을 읽는다.
#         (수진님 방침: 논문용으로 다듬을 때 데이터 목록·데이터도 모델 저장소로 모은다 — 그때 이 도구들도 함께 옮긴다)
# [2026-11-17] 이 도구를 Mark5.2 에서 모델 저장소(KAMPA)의 preprocessing/ 로 옮겼다. 재현 기록·mn10 은 같은 저장소에서 읽는다.
REPRO_DIR = os.path.join(os.path.dirname(BASE_DIR), "reproducibility", "external_sources")
GPT_CANDIDATES = os.path.join(REPRO_DIR, "selection", "external_metadata_candidates_20261022.csv")
# 샤워: GPT 목록(제목에 shower) 밖에서 FSD50K 메타데이터로 더 찾은 것. 제목·라벨을 읽고 실제 샤워 물소리만 남겼다.
FSD_SHOWER_EXTRA = [404087, 419233, 250785, 327844, 110329, 7993, 326201, 379156, 85900, 327845, 95700, 327450,
                    427125, 409129, 125173, 85756, 85755, 59237, 377889, 167118, 369869]
CLOTHO_SHOWER = {
    "evaluation": ["ShowerAndSoap.wav", "shower taking.wav", "Turning on Shower 2_1-2.wav", "Shower Running 01.wav",
                   "Shower and walk - front.wav", "RBH_Household_shower 03.wav"],
    "validation": ["Douche aan.wav"],
    "development": ["showering shower different intensities water bathroom WC.wav",
                    "Shower turning on and off with drain noise.wav", "Whistling in the shower.wav",
                    "Turning on Shower_1-2.wav", "Hotel room shower.wav", "Getting Into Shower and Turning On.wav",
                    "Shower, A.wav", "Bathroom shower.wav", "JM_HOME&amp;OFFICE_Shower 01 - Taking a shower.wav",
                    "Shower 2.wav", "Shower 1.wav", "untitled_shower 2.wav", "Shower Running 02.wav", "Shower faucet.wav"],
}
BBC_SHOWER = {"07019091": "Shower running.", "07067074": "Shower running.", "07019092": "Shower, person under shower.",
              "07027225": "Household: Shower, washing under it.", "07067075": "Person under shower.",
              "07027224": "Household: Shower turned on, run and turned off."}
SNU_CD_FOLDERS = {"027": "1F00m", "028": "1F06m", "029": "1F12m", "030": "2F00m", "031": "2F06m", "032": "2F12m",
                  "033": "3F00m", "034": "3F06m", "035": "3F12m"}
DUP_CORR = 0.98
# 물소리 클래스는 RMS 가 가장 큰 3초를 쓰므로, 물 아닌 사건(커튼 삐걱·문·말소리 등)이 같이 든 녹음은 그 사건이 창으로 뽑힌다.
# 2026-10-23 청취에서 FSD50K 257434(Squeak·Thump_and_thud, 태그 curtain)가 샤워로 들리지 않아 이 규칙을 넣었다.
# FSD50K 라벨이 아래 물소리 라벨 밖에 하나라도 있으면 뺀다('Tap' 은 수도꼭지가 아니라 두드림이다). Clotho 는 라벨이 없어 제목으로 거른다.
WATER_LABELS = {"Bathtub_(filling_or_washing)", "Water_tap_and_faucet", "Domestic_sounds_and_home_sounds", "Water", "Liquid",
                "Drip", "Pour", "Trickle_and_dribble", "Gurgling", "Splash_and_splatter", "Sink_(filling_or_washing)",
                "Toilet_flush", "Fill_(with_liquid)", "Stream", "Boiling"}
CLOTHO_TITLE_NONWATER = re.compile(r"walk|whistl", re.I)
LISTEN_REJECT = {("fsd50k", "257434"): "청취 탈락(2026-10-23, 샤워로 들리지 않음)"}
# 내용 검사에 걸렸지만 사람이 들어 보고 맞다고 한 것(내용 검사보다 우선). 185061 = 비행기 진공 변기(AudioSet 이 물소리로 못 봄).
LISTEN_ACCEPT = {("fsd50k", "185061"): "청취 통과(2026-10-23, 내용 검사 오탐)"}

# 내용 검사(2026-10-23): 게이트는 길이·음량·클리핑만 보므로 '소리는 멀쩡한데 내용이 다른' 클립을 못 거른다.
# AudioSet 으로 사전학습된 mn10(이 저장소 weights/mn10_as_mAP_471.pt, 527 라벨)으로 3초 창을 태깅해
#   물 점수   = 아래 물소리 라벨 확률의 최댓값
#   비물 점수 = 그 밖 라벨(방 울림 라벨 제외) 확률의 최댓값
# 을 잰다. 기준은 같은 클래스 교사 train 의 AI Hub 원본(사람이 단 라벨)에서 95%가 통과하는 값(5 백분위)으로 정한다.
# 물 점수 < 기준이거나 (물 − 비물) < 기준이면 '내용 탈락'. 3.3초 이하 입력은 이 모델의 527 출력이 포화되므로(ESC-50 로 확인)
# 3초를 10초로 반복해 넣는다. 가구 끌기는 AudioSet 에 맞는 라벨이 없어(기준 양성도 Music·Vehicle 로 흩어짐) 검사하지 않는다.
AS_WATER = ["Water", "Water tap, faucet", "Toilet flush", "Bathtub (filling or washing)", "Sink (filling or washing)", "Liquid",
            "Drip", "Pour", "Trickle, dribble", "Gush", "Splash, splatter", "Fill (with liquid)", "Stream", "Gurgling", "Slosh",
            "Rain", "Raindrop", "Rain on surface", "Waterfall"]
AS_DOG = ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Domestic animals, pets", "Animal"]

AS_ROOM = {"Inside, small room", "Inside, large room or hall", "Domestic sounds, home sounds"}
AS_LABELS_CSV = "audioset_meta/class_labels_indices.csv"     # EXT_ROOT 아래(EfficientAT a425fdce 의 metadata 와 같은 파일)

# dog_bark(2026-10-23 추가): 교사 train 781 중 AI Hub 525 를 Mark5.3 이 안 쓴 녹음으로만 채우면 413 까지라, FSD50K 의 개 소리
# (라벨 Dog 또는 Bark) 중 아직 안 쓴 것을 후보로 둔다. 말·음악·차량 라벨이 같이 붙은 녹음은 처음부터 뺀다.
FSD_DOG_LABELS = {"Dog", "Bark"}
FSD_DOG_EXCLUDE = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Music", "Vehicle"}
REF_PCT = 5
# talking(2026-10-23 추가): 교사 talking 양성이 Mark5.3 과 같은 CHiME·SINS 세션이라, train 을 말소리 반(FSD50K) + TV 반(VOiCES rm4 tele)으로
# 바꾼다(수진님 결정 2번 (나)). FSD50K 말소리는 음악·개·물·차량 라벨이 없는 것을 업로더마다 돌아가며 450개(시드 42).
# VOiCES 는 TV 방해음 하나를 마이크 10개로 받은 55분 녹음이라 녹음 1개로 친다 — 시간이 겹치지 않게 약 11초 간격 300구간, 마이크를 돌려 가며.
# [2026-10-25] talking = 어른 말소리 + TV(수진님 결정 B안). 아이 말소리는 talking 후보에서 빼고(라벨이 함께 붙은 녹음도 제외),
#              집 배경 others 에서도 계속 뺀다(FSD_HOME_EXCLUDE). 10-23 gate_report.csv 는 아이 말소리가 들어 있던 판의 결과다.
FSD_CHILD_LABELS = {"Child_speech_and_kid_speaking"}
FSD_SPEECH_LABELS = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Conversation", "Chatter"}
FSD_SPEECH_EXCLUDE = {"Music", "Dog", "Bark", "Toilet_flush", "Water", "Vehicle"} | FSD_CHILD_LABELS
N_SPEECH_CAND = 450
VOICES_DIR = "/mnt/d/workspace/journal/datasets_raw/voices_raw/rm4/tele"
VOICES_N = 300                                  # 간격 = (가장 긴 파일 길이 − 12초) / 300 ≈ 10.96초
# [2026-10-25] 아이 말소리(Child speech · 아기 옹알이 Babbling)는 talking 내용 검사 라벨에서 뺀다. others 의 '타겟 닮은 소리' 검사
#              (AS_TARGETLIKE)에는 그대로 넣어 아이 말소리가 others 로도 들어가지 않게 한다. 10-24 content_check_20261024.json 은 넣은 판이다.
AS_CHILD = ["Child speech, kid speaking", "Babbling"]
AS_TALK = ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation",
           "Narration, monologue", "Chatter", "Hubbub, speech noise, speech babble", "Television", "Radio"]
# 집 배경 others(2026-10-23 추가, 수진님 결정 3번): Mark5.3 이 안 쓴 SINS 집 세션은 약 37클립뿐이라 나머지를 FSD50K 집 생활음
# (SINS 의 cooking·eating·working 에 해당)으로 채운다. 음성 예제이므로 타겟 9종과 닮은 라벨이 붙은 녹음은 처음부터 뺀다.
FSD_HOME = {"cooking": {"Frying_(food)", "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Microwave_oven"},
            "eating": {"Chewing_and_mastication", "Crushing"},
            "working": {"Typing", "Computer_keyboard", "Writing", "Scissors", "Crumpling_and_crinkling", "Tearing", "Printer",
                        "Packing_tape_and_duct_tape"}}
FSD_HOME_N = {"cooking": 230, "eating": 100, "working": 130}   # cooking 130→230(10-23: train cooking 21 부족, 수진님 "더 받아야지")
FSD_HOME_EXCLUDE = FSD_SPEECH_LABELS | FSD_CHILD_LABELS | {"Music", "Vehicle", "Water", "Liquid", "Pour", "Toilet_flush", "Bathtub_(filling_or_washing)",
                                        "Sink_(filling_or_washing)", "Water_tap_and_faucet", "Drip", "Trickle_and_dribble", "Dog", "Bark",
                                        "Animal", "Door", "Knock", "Slam", "Thump_and_thud", "Hammer", "Drill", "Power_tool", "Sawing",
                                        "Tools", "Engine", "Mechanisms", "Walk_and_footsteps", "Boiling", "Gurgling", "Splash_and_splatter"}
# 음성 예제의 내용 검사는 반대 방향이다: 타겟 9종과 닮은 AudioSet 라벨 확률의 최댓값이, 지금 배경으로 쓰는 SINS 집 클립(사람이 단
# 활동 라벨)의 95 백분위보다 크면 뺀다.
AS_TARGETLIKE = AS_WATER + AS_DOG + AS_TALK[:-2] + AS_CHILD + ["Music", "Vehicle", "Car", "Drill", "Hammer", "Power tool", "Jackhammer",
                                                     "Sawing", "Thump, thud", "Knock", "Door", "Slam", "Walk, footsteps", "Run"]
# 클래스별 내용 검사 설정: 라벨 집합 · 기준 양성(버전, 클래스, 출처) · present(그 소리가 있어야 함) / absent(타겟 닮은 소리가 없어야 함)
CONTENT = {
    "water_toilet": dict(labels=AS_WATER, ref=("mark4.6", "water_toilet", ("aihub",), ("train",)), mode="present"),
    "water_shower": dict(labels=AS_WATER, ref=("mark4.7", "water_shower", ("aihub",), ("train",)), mode="present"),
    "dog_bark": dict(labels=AS_DOG, ref=("mark4.8", "dog_bark", ("aihub",), ("train",)), mode="present"),
    # [10-24] 기준 split train → val·test: 10-23 재배치로 talking train 에 CHiME·SINS 가 없어져(FSD50K·VOiCES) 기준 클립이 0개가 됐다.
    #         지금 CHiME·SINS talking 은 val·test(Mark5.3 학습 쪽 세션)에만 있다. 10-23 실행 결과는 gate_report.csv 그대로 둔다.
    "talking": dict(labels=AS_TALK, ref=("mark4.5", "talking", ("sins", "chime_home"), ("val", "test")), mode="present"),
    "others_house": dict(labels=AS_TARGETLIKE, ref=("mark4.1", "others", ("sins",), ("train", "val", "test")), mode="absent"),
    # [10-24] Mark4.5 전용 others: SINS 평가 세트 집 생활음 · FSD50K 사람 비발화음. 음성 예제라 others_house 와 같은 기준(absent).
    "others_house_sins": dict(labels=AS_TARGETLIKE, ref=("mark4.1", "others", ("sins",), ("train", "val", "test")), mode="absent"),
    "others_human": dict(labels=AS_TARGETLIKE, ref=("mark4.1", "others", ("sins",), ("train", "val", "test")), mode="absent"),
}

# talking 재수집(2026-10-24, 수진님 결정 나안): 10-23 talking train(FSD50K 말소리 + VOiCES TV)으로 학습한 mn10 교사가 1에폭 뒤로 val 이
# 계속 떨어졌다(train 0.85→0.9955, val 0.9605→0.92). val 은 CHiME·SINS 집 녹음이라 train 을 집 녹음 쪽으로 바꾼다.
#   SINS 평가 세트(DCASE2018 Task5 eval, Zenodo 1964758, CC BY-NC-ND 4.0) — 논문(Dekkers 2018 §2.3)상 개발 세트와 다른 시간대(1/3).
#     Mark5.3·교사가 쓰는 SINS 는 모두 개발 세트(DevNode)라 녹음이 겹치지 않는다. 10초 구간 → RMS 최대 3초.
#     watching_tv·social_activity → talking, cooking·eating·working → others_house_sins (Mark4.5 others 의 집 배경 자리)
#   CHiME-6(OpenSLR 150, CC BY-SA 4.0) dev S02·S09 · eval S01·S21 — 실제 가정 저녁 모임. 전사의 말소리 구간(겹침 합침) 중 9.2초 이상인
#     구간에서 3초(게이트에서 떨어지면 +3·+6초 — 같은 구간 안), 세션마다 고르게, Kinect 배열(U01~U06 CH1)을 돌려 가며.
#   FSD50K 사람 비발화음(숨·기침·재채기·한숨·트림·씹기·호흡) — 말소리·웃음·울음·음악 라벨이 없는 것, 업로더마다 1개(시드 42, 400개).
SINS_EVAL_DIR = "dcase2018t5/eval_audio"                    # EXT_ROOT 아래
SINS_EVAL_PICK = "dcase2018t5/pick_eval.json"
SINS_EVAL_CLS = {"watching_tv": ("talking", "tv"), "social_activity": ("talking", "social"),
                 "cooking": ("others_house_sins", "cooking"), "eating": ("others_house_sins", "eating"), "working": ("others_house_sins", "working")}
CHIME6_SESSIONS = {"S02": "dev", "S09": "dev", "S01": "eval", "S21": "eval"}
CHIME6_PER_SESSION = 80
CHIME6_MIN_REGION = 9.2
FSD_HUMAN_LIST = "tools/ids_human_nonspeech_all.txt"        # 고른 400개(split,id) — 받을 때 쓴 목록의 전체판
FSD_HUMAN_LABELS = ["Breathing", "Cough", "Sneeze", "Sigh", "Burping_and_eructation", "Chewing_and_mastication", "Respiratory_sounds"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def used_ids():
    import pandas as pd
    prov = pd.read_excel(R.PROV_PATH)
    fsd = {str(int(x)) for x in prov["fsd50k_fname"].dropna()}
    esc = set()
    for col in ("aihub_src_file", "local_filename", "aug_source_file"):
        for f in prov[col].dropna().astype(str):
            m = re.match(r"^\d-(\d+)-[A-Z]-\d+", os.path.basename(f))
            if m:
                esc.add(m.group(1))
    return fsd | esc


def candidates(ext):
    meta, labels = {}, {}
    for sp in ("dev", "eval"):
        for k, v in json.load(open(f"{ext}/fsd50k_meta/FSD50K.metadata/{sp}_clips_info_FSD50K.json")).items():
            meta[k] = v
        for r in csv.DictReader(open(f"{ext}/fsd50k_meta/FSD50K.ground_truth/{sp}.csv")):
            labels[r["fname"]] = r["labels"]
    out = []
    gpt = list(csv.DictReader(open(GPT_CANDIDATES, encoding="utf-8-sig")))
    fsd_rows = [(r["class"], r["freesound_id"]) for r in gpt] + [("water_shower", str(i)) for i in FSD_SHOWER_EXTRA]
    for cls, fid in fsd_rows:
        v = meta[fid]
        out.append(dict(cls=cls, source="fsd50k", ext_id=fid, path=f"{ext}/fsd50k_audio/{fid}.wav", group=v["uploader"],
                        license=v["license"], title=v["title"], url=f"https://freesound.org/s/{fid}/", freesound_id=fid,
                        labels=labels[fid]))
    used = used_ids()
    for fid in sorted(labels, key=int):
        L = set(labels[fid].split(","))
        if L & FSD_DOG_LABELS and not L & FSD_DOG_EXCLUDE and fid not in used:
            v = meta[fid]
            out.append(dict(cls="dog_bark", source="fsd50k", ext_id=fid, path=f"{ext}/fsd50k_audio/{fid}.wav", group=v["uploader"],
                            license=v["license"], title=v["title"], url=f"https://freesound.org/s/{fid}/", freesound_id=fid,
                            labels=labels[fid]))
    def rr_uploader(fids, n):
        g = {}
        for fid in sorted(fids, key=int):
            g.setdefault(meta[fid]["uploader"], []).append(fid)
        rng = random.Random(42)
        ups = sorted(g)
        for u in ups:
            rng.shuffle(g[u])
        got = []
        while len(got) < n and any(g[u] for u in ups):
            for u in ups:
                if g[u] and len(got) < n:
                    got.append(g[u].pop(0))
        return got

    def fsd_row(cls, fid, extra=None):
        v = meta[fid]
        d = dict(cls=cls, source="fsd50k", ext_id=fid, path=f"{ext}/fsd50k_audio/{fid}.wav", group=v["uploader"],
                 license=v["license"], title=v["title"], url=f"https://freesound.org/s/{fid}/", freesound_id=fid, labels=labels[fid])
        d.update(extra or {})
        return d
    sets = {fid: set(labels[fid].split(",")) for fid in labels}
    speech = [f for f, L in sets.items() if L & FSD_SPEECH_LABELS and not L & FSD_SPEECH_EXCLUDE and f not in used]
    out += [fsd_row("talking", f) for f in rr_uploader(speech, N_SPEECH_CAND)]
    for act, labs in FSD_HOME.items():
        pool = [f for f, L in sets.items() if L & labs and not L & FSD_HOME_EXCLUDE and f not in used
                and not any(L & FSD_HOME[a] for a in FSD_HOME if list(FSD_HOME).index(a) < list(FSD_HOME).index(act))]
        out += [fsd_row("others_house", f, {"activity": act}) for f in rr_uploader(pool, FSD_HOME_N[act])]
    vfiles = sorted(glob.glob(os.path.join(VOICES_DIR, "*.wav")))
    vdur = {f: sf.info(f).duration for f in vfiles}
    step = (max(vdur.values()) - 12.0) / VOICES_N
    for k in range(VOICES_N):
        t = round(k * step, 2)
        usable = [f for f in vfiles if vdur[f] >= t + 12.0]
        f = usable[k % len(usable)]
        mic = re.search(r"tele-(mc\d+-[a-z]+-[a-z0-9]+)", f).group(1)
        out.append(dict(cls="talking", source="voices", ext_id=f"rm4-tele_{mic}_t{int(t):04d}", path=f, group=mic, start=t,
                        license="CC BY 4.0", title=f"VOiCES rm4 tele distractor ({mic}, {int(t)}s)",
                        url="https://iqtlabs.github.io/voices/", freesound_id=""))
    for split, names in CLOTHO_SHOWER.items():
        md = {r["file_name"]: r for r in csv.DictReader(open(f"{ext}/clotho_meta/clotho_metadata_{split}.csv", encoding="latin-1"))}
        for n in names:
            r = md[n]
            p = glob.glob(f"{ext}/clotho/audio/{split}/**/{glob.escape(n)}", recursive=True)
            out.append(dict(cls="water_shower", source="clotho", ext_id=f"{split}:{n}", path=p[0] if p else "", group=r["manufacturer"],
                            license=r["license"], title=n, url=r["sound_link"], freesound_id=str(r["sound_id"])))
    for f in sorted(glob.glob(f"{ext}/bedavin/*.wav")):
        c, aid = os.path.basename(f)[:-4].split("__")
        out.append(dict(cls={"shower": "water_shower", "toilet": "water_toilet"}[c], source="bedavin", ext_id=aid, path=f,
                        group="bedavin", license="CC BY 4.0", title=c, url="https://zenodo.org/records/14486235", freesound_id=""))
    for bid, t in BBC_SHOWER.items():
        out.append(dict(cls="water_shower", source="bbc", ext_id=bid, path=f"{ext}/bbc_sfx/{bid}.wav", group="bbc",
                        license="RemArc", title=t, url=f"https://sound-effects.bbcrewind.co.uk/search?q={bid}", freesound_id=""))
    for folder, pos in SNU_CD_FOLDERS.items():
        for f in sorted(glob.glob(f"{ext}/snu_b36_50e/cd_wav/{folder}*.wav")):
            out.append(dict(cls="dragging", source="snu_b36_50e", ext_id=os.path.basename(f)[:-4], path=f, group=f"snu_{pos}",
                            license="GPL-3.0", title=f"CD-{pos}", url="https://zenodo.org/records/3257142", freesound_id=""))
    # ── [10-24] talking 재수집 ──
    pick_path = f"{ext}/{SINS_EVAL_PICK}"
    if not os.path.isfile(pick_path):                       # [10-25] 외부 자료 폴더에 없으면 저장소 사본
        pick_path = os.path.join(REPRO_DIR, "selection", os.path.basename(SINS_EVAL_PICK))
    if os.path.isfile(pick_path):
        for p in json.load(open(pick_path)):
            cls, act = SINS_EVAL_CLS[p["cls"]]
            out.append(dict(cls=cls, source="sins_eval", ext_id=f"ex{p['ses']}_{p['seg']}_n{p['node']}",
                            path=f"{ext}/{SINS_EVAL_DIR}/{os.path.basename(p['file'])}", group=f"s{p['ses']}", activity=act,
                            license="CC BY-NC-ND 4.0", title=os.path.basename(p["orig"]), url="https://zenodo.org/records/1964758",
                            freesound_id=""))
    out += chime6_candidates(ext)
    hl = f"{ext}/{FSD_HUMAN_LIST}"
    if not os.path.isfile(hl):                              # [10-25] 외부 자료 폴더에 없으면 저장소 사본
        hl = os.path.join(REPRO_DIR, "lists", os.path.basename(FSD_HUMAN_LIST))
    if os.path.isfile(hl):
        for line in open(hl):
            sp, fid = line.strip().split(",")
            hit = [x for x in FSD_HUMAN_LABELS if x in labels[fid].split(",")]
            out.append(fsd_row("others_human", fid, {"activity": hit[0] if hit else ""}))
    return out


def chime6_candidates(ext):
    """CHiME-6 전사의 말소리 구간에서 세션마다 고르게 시작 시각을 고른다(Kinect 배열을 돌려 가며)."""
    def t(s):
        h, m, x = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(x)
    out = []
    for ses, part in CHIME6_SESSIONS.items():
        jp = glob.glob(f"{ext}/chime6_meta/**/transcriptions/{part}/{ses}.json", recursive=True)
        wavs = sorted(glob.glob(f"{ext}/chime6_audio/**/{ses}_U0?.CH1.wav", recursive=True))
        if not jp or not wavs:
            continue
        iv = sorted((t(u["start_time"]), t(u["end_time"])) for u in json.load(open(jp[0])))
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        regions = [r for r in merged if r[1] - r[0] >= CHIME6_MIN_REGION]
        k = min(CHIME6_PER_SESSION, len(regions))
        idx = sorted({round(i * (len(regions) - 1) / max(1, k - 1)) for i in range(k)})
        for j, i in enumerate(idx):
            a, b = regions[i]
            w = wavs[j % len(wavs)]
            arr = re.search(r"_(U0\d)\.CH1", w).group(1)
            start = round(a + 0.1, 2)
            out.append(dict(cls="talking", source="chime6", ext_id=f"{ses}_{arr}_t{int(start):05d}", path=w, group=ses, start=start,
                            activity="speech", license="CC BY-SA 4.0", title=f"CHiME-6 {ses} {arr} {start:.1f}s",
                            url="https://openslr.org/150", freesound_id=""))
    return out


def pick_window(path):
    """첫 채널을 16kHz 로 바꾸고 RMS 순으로 창을 본다. (seg, start, rank, why)"""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    y = slice_resample(x, sr, 0.0, len(x) / sr)
    n, hop = int(SEG_SEC * TARGET_SR), int(0.1 * TARGET_SR)
    if len(y) < n:
        return None, None, None, f"길이부족 {len(y)/TARGET_SR:.2f}s"
    starts = np.arange(0, len(y) - n + 1, hop)
    c = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    rms = np.sqrt((c[starts + n] - c[starts]) / n)
    order = np.argsort(-rms)
    why = "?"
    for rank, i in enumerate(order):
        if rms[i] < 0.5 * rms[order[0]]:
            break
        seg = y[starts[i]:starts[i] + n]
        bad = check_waveform(seg, TARGET_SR, min_duration=SEG_SEC)
        if bad is None:
            return seg, starts[i] / TARGET_SR, rank, ""
        if rank == 0:
            why = bad
    return None, None, None, why


class ContentTagger:
    """mn10 AudioSet 태거. tag(파일 목록) → [N, 527] 확률."""

    def __init__(self, ext):
        import io, contextlib, torch
        third = os.path.join(os.path.dirname(BASE_DIR), "third_party", "efficientat")
        if third not in sys.path:
            sys.path.insert(0, third)
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
        self.weights = os.path.join(os.path.dirname(BASE_DIR), "weights", "mn10_as_mAP_471.pt")
        with contextlib.redirect_stdout(io.StringIO()):
            self.net = get_model(num_classes=527, pretrained_name=None, width_mult=1.0, head_type="mlp", se_dims="c")
            self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0, timem=0)
        self.net.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True), strict=True)
        self.net.eval(); self.mel.eval()
        self.names = [r["display_name"] for r in csv.DictReader(open(os.path.join(ext, AS_LABELS_CSV)))]
        assert len(self.names) == 527
        self.idx = {}
        for cls, cfg in CONTENT.items():
            ti = [self.names.index(n) for n in cfg["labels"]]
            self.idx[cls] = (ti, [i for i, n in enumerate(self.names) if i not in ti and n not in AS_ROOM])

    def tag(self, paths):
        import torch, torchaudio
        out = []
        for i in range(0, len(paths), 32):
            xs = []
            for p in paths[i:i + 32]:
                y, sr = sf.read(p, dtype="float32", always_2d=True)
                y = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(y[:, 0])), sr, 32000)[:96000]
                xs.append(y.repeat(4)[:320000])                  # 3초 → 10초 반복
            with torch.no_grad():
                logit, _ = self.net(self.mel(torch.stack(xs)).unsqueeze(1))
            out.append(torch.sigmoid(logit).reshape(len(xs), 527).numpy())
        return np.concatenate(out)

    def scores(self, P, cls):
        """목표 점수(그 클래스 라벨 확률의 최댓값) · 그 밖 점수 · 그 밖에서 가장 높은 라벨."""
        ti, ni = self.idx[cls]
        w, o = P[:, ti].max(1), P[:, ni].max(1)
        top = [self.names[ni[j]] for j in P[:, ni].argmax(1)]
        top_t = [self.names[ti[j]] for j in P[:, ti].argmax(1)]
        return w, o, top, top_t


def reference_thresholds(tagger, only=None):
    """기준 클립(사람이 단 라벨)에서 기준값을 잰다.
    present: 목표 점수·(목표−그 밖) 의 5 백분위(이보다 낮으면 탈락) / absent: 타겟 닮은 점수의 95 백분위(이보다 높으면 탈락).
    only: 이 클래스들만 잰다([10-24] 후보가 있는 클래스만)."""
    import pandas as pd
    from cross_reject_experts import data_path
    prov = pd.read_excel(R.PROV_PATH)
    act = prov[prov["removed_20260715"].astype(str) == "active"]
    thr = {}
    for cls, cfg in CONTENT.items():
        if only is not None and cls not in only:
            continue
        ver, tcls, srcs, splits = cfg["ref"]
        o = act[(act["mark_version"] == ver) & (act["source_type"] == "original") & (act["target_class"] == tcls)
                & (act["assigned_split"].isin(splits)) & (act["source"].isin(srcs))]
        if tcls == "others":                                    # 배경만(다른 타겟 자리는 빼고)
            o = o[~o["mark52_note"].astype(str).str.startswith(("cross_target", "hardneg"))]
        w, oth, _, _ = tagger.scores(tagger.tag([data_path(ver, r.assigned_split, r.local_filename) for r in o.itertuples()]), cls)
        if cfg["mode"] == "present":
            thr[cls] = dict(mode="present", n_ref=len(w), target=float(np.percentile(w, REF_PCT)),
                            margin=float(np.percentile(w - oth, REF_PCT)), target_median=float(np.median(w)))
            print(f"[내용 기준] {cls}: 기준 클립 {len(w)} · 목표 점수 ≥ {thr[cls]['target']:.3f} · (목표−그 밖) ≥ {thr[cls]['margin']:.3f}", flush=True)
        else:
            thr[cls] = dict(mode="absent", n_ref=len(w), targetlike_max=float(np.percentile(w, 100 - REF_PCT)),
                            targetlike_median=float(np.median(w)))
            print(f"[내용 기준] {cls}: 기준 클립 {len(w)} · 타겟 닮은 점수 ≤ {thr[cls]['targetlike_max']:.3f}", flush=True)
    return thr


def pick_fixed(path, start):
    """정해진 시각(VOiCES)에서 3초. 게이트에서 떨어지면 3초씩 미뤄 두 번 더 본다(같은 11초 칸 안이라 다른 구간과 안 겹친다)."""
    info = sf.info(path)
    why = "?"
    for off in (0.0, 3.0, 6.0):
        a = int((start + off) * info.samplerate)
        x, sr = sf.read(path, start=a, frames=int(SEG_SEC * info.samplerate), dtype="float32", always_2d=True)
        seg = slice_resample(np.ascontiguousarray(x[:, 0]), sr, 0.0, len(x) / sr)
        bad = check_waveform(seg, TARGET_SR, min_duration=SEG_SEC)
        if bad is None:
            return seg, start + off, int(off // 3), ""
        why = bad
    return None, None, None, why


def logmel(seg):
    import torch, torchaudio
    m = torchaudio.transforms.MelSpectrogram(TARGET_SR, n_fft=1024, hop_length=256, n_mels=64)(torch.from_numpy(seg))
    v = torch.log(m + 1e-6).flatten().numpy()
    return (v - v.mean()) / (v.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_root", default=EXT_ROOT_DEFAULT)
    ap.add_argument("--sources", default="", help="[10-24] 쉼표로 출처를 고르면 그 후보만 본다(예: sins_eval,chime6,fsd50k)")
    ap.add_argument("--classes", default="", help="[10-24] 쉼표로 클래스를 고르면 그 후보만 본다(예: talking,others_human)")
    ap.add_argument("--tag", default="", help="[10-24] 보고서 이름 꼬리표(gate_report_<tag>.csv) — 10-23 보고서를 덮어쓰지 않게")
    a = ap.parse_args()
    ext = a.ext_root
    used = used_ids()
    cands = candidates(ext)
    if a.sources:
        cands = [c for c in cands if c["source"] in a.sources.split(",")]
    if a.classes:
        cands = [c for c in cands if c["cls"] in a.classes.split(",")]
    sfx = f"_{a.tag}" if a.tag else ""
    seen_fs = {c["freesound_id"] for c in cands if c["source"] == "fsd50k"}
    rows, segs = [], {}
    for c in cands:
        r = dict(c, status="", why="", seg_start_sec="", window_rank="", out_path="", sha256="")
        r.setdefault("labels", "")
        nonwater = sorted(set(r["labels"].split(",")) - WATER_LABELS) if r["labels"] else []
        if (c["source"], c["ext_id"]) in LISTEN_REJECT:
            r.update(status="제외", why=LISTEN_REJECT[(c["source"], c["ext_id"])])
        elif c["cls"].startswith("water_") and nonwater:
            r.update(status="제외", why="물 아닌 소리 라벨: " + ",".join(nonwater))
        elif c["source"] == "clotho" and CLOTHO_TITLE_NONWATER.search(c["title"]):
            r.update(status="제외", why="제목에 물 아닌 소리(걷기·휘파람)")
        elif c["freesound_id"] and c["freesound_id"] in used:
            r.update(status="제외", why="이미 쓴 Freesound 원본")
        elif c["source"] == "clotho" and c["freesound_id"] in seen_fs:
            r.update(status="제외", why="FSD50K 와 같은 Freesound 원본")
        elif not c["path"] or not os.path.exists(c["path"]):
            r.update(status="없음", why="오디오 파일 없음")
        else:
            seg, st, rank, why = pick_fixed(c["path"], c["start"]) if "start" in c else pick_window(c["path"])
            if seg is None:
                r.update(status="탈락", why=why)
            else:
                op = f"{ext}/gated/{c['cls']}/{c['source']}_{re.sub(r'[^0-9A-Za-z_-]+', '_', c['ext_id'])}.wav"
                os.makedirs(os.path.dirname(op), exist_ok=True)
                sf.write(op, seg, TARGET_SR, subtype="FLOAT")
                r.update(status="통과", seg_start_sec=round(float(st), 2), window_rank=rank, out_path=op, sha256=sha(op))
                segs[op] = logmel(seg)
        rows.append(r)
    # 내용 검사(물소리·개 클래스)
    tagger = ContentTagger(ext)
    thr = reference_thresholds(tagger, {r["cls"] for r in rows})
    for cls in thr:
        chk = [r for r in rows if r["status"] == "통과" and r["cls"] == cls]
        if not chk:
            continue
        w, o, top, topt = tagger.scores(tagger.tag([r["out_path"] for r in chk]), cls)
        t = thr[cls]
        for r, wi, oi, ti, tt in zip(chk, w, o, top, topt):
            r.update(as_target=round(float(wi), 3), as_other=round(float(oi), 3), as_other_top=ti, as_target_top=tt)
            if (r["source"], r["ext_id"]) in LISTEN_ACCEPT:
                r.update(why=LISTEN_ACCEPT[(r["source"], r["ext_id"])])
            elif t["mode"] == "present" and (wi < t["target"] or wi - oi < t["margin"]):
                r.update(status="내용탈락", why=f"목표 점수 {wi:.2f}(기준 {t['target']:.3f}) · 그 밖 {oi:.2f}({ti})")
            elif t["mode"] == "absent" and wi > t["targetlike_max"]:
                r.update(status="내용탈락", why=f"타겟 닮은 점수 {wi:.2f}({tt}, 기준 ≤ {t['targetlike_max']:.3f})")
    for r in rows:
        for k in ("as_target", "as_target_top", "as_other", "as_other_top"):
            r.setdefault(k, "")
    # 같은 녹음 의심 쌍(클래스 안)
    dup = {}
    for cls in sorted({r["cls"] for r in rows}):
        ps = [r["out_path"] for r in rows if r["cls"] == cls and r["status"] == "통과"]
        if len(ps) < 2:
            continue
        M = np.stack([segs[p] for p in ps])
        C = M @ M.T / M.shape[1]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if C[i, j] >= DUP_CORR:
                    dup.setdefault(ps[j], (ps[i], float(C[i, j])))
    for r in rows:
        if r["out_path"] in dup:
            first, cc = dup[r["out_path"]]
            r["why"] = f"같은 녹음 의심: {os.path.basename(first)} (상관 {cc:.3f})"
            r["status"] = "중복"
    rep = f"{ext}/gated/gate_report{sfx}.csv"
    with open(rep, "w", encoding="utf-8-sig", newline="") as f:
        cols = list(dict.fromkeys(k for r in rows for k in r))         # 행마다 열이 다를 수 있다(activity·start 등)
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    json.dump({"content_thresholds": thr, "tagger_weights": os.path.relpath(tagger.weights, R.CODES_ROOT),
               "tagger_weights_sha256": sha(tagger.weights), "ref_percentile": REF_PCT, "content": {k: dict(v, ref=list(v["ref"])) for k, v in CONTENT.items()}},
              open(f"{ext}/gated/content_check{sfx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[보고서] {rep}")
    cnt = Counter((r["cls"], r["source"], r["status"]) for r in rows)
    for k in sorted(cnt):
        print(f"  {k[0]:13s} {k[1]:12s} {k[2]:4s} {cnt[k]}")
    for cls in sorted({r["cls"] for r in rows}):
        ok = [r for r in rows if r["cls"] == cls and r["status"] == "통과"]
        print(f"[통과] {cls:13s} {len(ok)}개 · 묶음(업로더·위치) {len({r['group'] for r in ok})}")


if __name__ == "__main__":
    main()
