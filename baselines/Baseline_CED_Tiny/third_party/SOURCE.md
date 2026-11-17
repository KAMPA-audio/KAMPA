# 출처
- 모델·원격 코드: https://huggingface.co/mispeech/ced-tiny (Apache-2.0), revision ace276d29dd0bb3f3517b0fa8cf300738c409019  (2026-09-05 스냅샷)
- weights/ced-tiny/ 에 그 스냅샷 사본(config.json · *_ced.py · model.safetensors · preprocessor_config.json)을 두었다.
  model_def.py 는 이 폴더가 있으면 허브 접속 없이 여기서 로드한다(trust_remote_code=True).
- GitHub 저장소 RicherMans/CED 의 학습 코드는 GPL-3.0 이라 **가져오지 않았다** (조건표 7절 ⑬).
- 논문: Dinkel et al., "CED: Consistent Ensemble Distillation for Audio Tagging," ICASSP 2024.

## 스냅샷 파일 지문 (2026-09-05, revision ace276d29dd0bb3f3517b0fa8cf300738c409019)
- config.json        sha256 3a060c57c28b8138cf66bbab2c0dcab79f07c871b600f52a06f2652bf3cad2bd
- model.safetensors  sha256 0e086f0cd62814c6def89001f3f25193f75955696f6975ef6800af31d00d6dd7
- modeling_ced.py    sha256 c8a8e8b865e4e3dcebffefb40215d6778bfa380ecea388f11806e33521a29327
파일이 갈아끼워지면 이 지문과 달라진다. model_def.py 는 로드한 config 의 mel 규격·패치 규격을 기대값과 대조해
다른 스냅샷이면 멈춘다.

## 주의 — 첫 실행에는 인터넷이 필요하다
- 원격 코드 configuration_ced.py 는 outputdim 이 527 이면 __init__ 에서 허브 `topel/ConvNeXt-Tiny-AT` 의
  `class_labels_indices.csv` 를 받는다. transformers 의 from_pretrained 는 config.json(outputdim 527)으로 먼저
  인스턴스를 만든 뒤 우리가 준 outputdim=9 를 덮어쓰고(from_dict), 로그 출력 때 기본값 인스턴스를 한 번 더
  만든다(to_diff_dict). 두 경로 모두 527 을 타므로 이 다운로드는 피할 수 없다. 한 번 받으면 HF 캐시에 남아
  그 뒤로는 오프라인(HF_HUB_OFFLINE=1)에서도 된다. 코랩은 온라인이라 문제없고, 스냅샷 사본(config.json 포함)은
  원본 그대로다.
- trust_remote_code=True 는 모델 저장소의 파이썬 파일(modeling_ced.py 등)을 그대로 실행한다는 뜻이다.
  로컬 스냅샷을 쓸 때는 우리가 위 지문으로 확인한 파일이 실행되고, 허브 경로일 때는 revision 을 고정해 받는다.
