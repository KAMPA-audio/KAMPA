"""[신설 2026-11-18] 클래스 이름을 바꾼 기록과, 옛 이름으로 적힌 파일을 읽을 때 새 이름으로 맞추는 곳.

talking → speech: 논문(영문 Speech · 국문 말소리)과 이름을 맞췄다. 클래스 정의(어른 말소리 + TV, 아이 말소리 제외, 2026-10-25)와
클래스 순서(다섯 번째)는 그대로다. 체크포인트에는 클래스 이름이 아니라 순서만 들어 있어 학생·교사 가중치는 그대로 맞는다.

이름을 바꾸지 않고 둔 것 — 읽는 곳에서 canonical_class_name()·canonical_row() 로 맞춘다.
  · 오디오 폴더·파일 이름(data_*/talking/, talking_train_001.wav): 출처 장부(data_provenance.xlsx)·추가실험 기록과 묶인 식별자
  · Mark4.5 교사 체크포인트의 provenance(classes ["talking", "others"] · target_class "talking"): 교사 저장소 쪽 이름
  · 끝난 실험의 결과 파일: 가짜 정답·합의·상한 라벨 CSV(label·label2), 교사 점수 캐시(p_talking), 예측 CSV(Prob_talking)
"""

LEGACY_CLASS_NAMES = {"talking": "speech"}

_LABEL_FIELDS = ("label", "label2", "True Label", "Predicted Label", "Raw Predicted Label")
_COLUMN_PREFIXES = ("PreBias_Prob_", "Prob_", "p_")


def canonical_class_name(name):
    """옛 클래스 이름이면 지금 이름을, 아니면 받은 값을 그대로 돌려준다(빈 문자열·None 포함)."""
    return LEGACY_CLASS_NAMES.get(name, name)


def canonical_row(row):
    """CSV 한 행(dict)에서 라벨 값(label·label2·True/Predicted Label)과 클래스 이름이 붙은 열 이름
    (p_talking · Prob_talking · PreBias_Prob_talking)을 새 이름으로 바꾼 새 dict 를 돌려준다. 다른 열은 그대로다."""
    out = {}
    for key, value in row.items():
        if isinstance(key, str):
            for prefix in _COLUMN_PREFIXES:
                if key.startswith(prefix):
                    key = prefix + canonical_class_name(key[len(prefix):])
                    break
        if key in _LABEL_FIELDS:
            value = canonical_class_name(value)
        out[key] = value
    return out
