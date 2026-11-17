"""Build deterministic Colab notebooks that unpack the reviewed src ZIP and the result aggregator from Drive by SHA-256."""
from pathlib import Path
import hashlib,io,json,zipfile
import nbformat as nbf
BASE=Path(__file__).resolve().parents[1]

def bundle():
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted((BASE/'src').rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.pyo'):continue
            item=zipfile.ZipInfo(p.relative_to(BASE/'src').as_posix(),date_time=(2026,11,14,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED;item.external_attr=0o644<<16
            archive.writestr(item,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return output.getvalue()

def build():
    payload=bundle();sha=hashlib.sha256(payload).hexdigest()
    agg_sha=hashlib.sha256((BASE/'notebooks/KAMPA_결과집계.py').read_bytes()).hexdigest()
    (BASE/'notebooks/KAMPA_추가실험_실행코드.zip').write_bytes(payload)
    notes=json.loads((BASE/'tools/notebook_notes.json').read_text())
    for name,description in notes.items():
        stage='1_3' if name.startswith('01') else '4'
        description=description.replace('같은 GPU 환경','같은 CPU 또는 GPU 환경').replace('별도의 GPU 런타임','별도의 런타임')
        description+='\n실행 환경: CPU를 지원합니다. GPU를 사용하면 L4를 우선 사용하고 Blackwell 계열은 기존 프로젝트 정책상 제외합니다. 환경별 결과 폴더가 자동 분리되며 서로 다른 환경의 결과를 합산하지 않습니다.\n\n입력 검증: 원본 FLOAT 음원과 실제 PCM_16 입력의 SHA-256을 별도로 확인합니다. 교사 8개·점수 캐시·전체 입력의 버전이 모두 일치할 때 기존 교사 점수를 재사용하며, 교사 재학습이나 재추론은 필요하지 않습니다.\n'
        cells=[]
        def md(s):cells.append(nbf.v4.new_markdown_cell(s))
        def code(s):cells.append(nbf.v4.new_code_cell(s))
        md(description)
        md('## 1. Google Drive 연결\n드라이브 `NewProject/codes/experiments/Mark5.3/feedback/20261114/notebooks/`의 이 노트북을 Colab에서 열고 \'모두 실행\'을 누르세요. GPU는 L4를 사용합니다. CPU로도 실행할 수 있습니다.')
        code('from google.colab import drive\ndrive.mount("/content/drive")')
        md('## 2. 환경과 실행 코드 준비\n실행 코드는 같은 폴더의 `KAMPA_추가실험_실행코드.zip`에서 풀고, 결과 집계는 같은 폴더의 `KAMPA_결과집계.py`를 씁니다. 두 파일 중 하나라도 이 노트북에 적힌 SHA-256과 다르면 멈춥니다. Torch 환경 검사는 별도의 Python 프로세스에서 수행합니다. 실행에 시간 제한을 두지 않습니다.')
        code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy>=1.24", "pandas>=2.0", "scipy>=1.11", "scikit-learn>=1.3", "soundfile>=0.12"], check=True)
# Colab이 제공한 서로 호환되는 torch/torchaudio/torchvision 조합을 유지합니다.
subprocess.run([sys.executable, "-c", "import torch,torchaudio,torchvision; print('torch:',torch.__version__,'torchaudio:',torchaudio.__version__,'torchvision:',torchvision.__version__); print('CUDA available:',torch.cuda.is_available())"], check=True)
''')
        code(f'''from pathlib import Path
import hashlib, io, json, zipfile
PROJECT_ROOT = Path("/content/drive/MyDrive/NewProject/codes/experiments/Mark5.3")
CODE_ZIP = PROJECT_ROOT / "feedback/20261114/notebooks/KAMPA_추가실험_실행코드.zip"
CODE_SHA256 = "{sha}"
if not CODE_ZIP.exists():
    raise SystemExit(f"실행 코드 묶음이 없습니다: {{CODE_ZIP}}")
archive = CODE_ZIP.read_bytes()
archive_sha = hashlib.sha256(archive).hexdigest()
if archive_sha != CODE_SHA256:
    raise SystemExit(f"실행 코드 묶음의 SHA-256이 노트북과 다릅니다: {{archive_sha[:12]}} (기대값 {{CODE_SHA256[:12]}})")
PACKAGE = Path("/content/kampa_feedback_20261114_v2_{sha[:12]}")
PACKAGE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(PACKAGE)
# 매번 묶음에서 복원하므로 이전 실행 코드가 섞이지 않습니다.
print("실행 코드 준비 완료:", PACKAGE, "SHA-256", CODE_SHA256[:12])
AGGREGATOR_FILE = PROJECT_ROOT / "feedback/20261114/notebooks/KAMPA_결과집계.py"
AGGREGATOR_SHA256 = "{agg_sha}"
if not AGGREGATOR_FILE.exists():
    raise SystemExit(f"결과 집계 파일이 없습니다: {{AGGREGATOR_FILE}}")
aggregator_bytes = AGGREGATOR_FILE.read_bytes()
aggregator_sha = hashlib.sha256(aggregator_bytes).hexdigest()
if aggregator_sha != AGGREGATOR_SHA256:
    raise SystemExit(f"결과 집계 파일의 SHA-256이 노트북과 다릅니다: {{aggregator_sha[:12]}} (기대값 {{AGGREGATOR_SHA256[:12]}})")
# 실행 코드 폴더 밖에 둡니다. 실행 코드의 .py 해시가 회차 식별값에 들어가므로 완료 회차가 그대로 인정됩니다.
AGGREGATOR = Path("/content/kampa_aggregate_{agg_sha[:12]}/KAMPA_결과집계.py")
AGGREGATOR.parent.mkdir(parents=True, exist_ok=True)
AGGREGATOR.write_bytes(aggregator_bytes)
print("결과 집계 준비 완료:", AGGREGATOR, "SHA-256", AGGREGATOR_SHA256[:12])
''')
        md('## 3. 실행 범위와 저장 위치\n기본은 N=[200, 2000]입니다. 두 라벨 수를 모두 실행하며, 같은 환경과 설정으로 이미 완료한 회차는 검증 후 건너뜁니다.')
        code(f'''probe = subprocess.run([sys.executable, "-c", "import json; from feedback_core import runtime_metadata; print(json.dumps(runtime_metadata(),sort_keys=True))"], cwd=PACKAGE, capture_output=True, text=True, check=True)
RUNTIME = json.loads(probe.stdout)
RUNTIME_ID = hashlib.sha256(json.dumps(RUNTIME,sort_keys=True).encode()).hexdigest()[:12]
OUTPUT_ROOT = PROJECT_ROOT.parent / "KAMPA_feedback_20261114_v2" / RUNTIME_ID / "{stage}"
LABEL_BUDGETS = [200, 2000]
TRAIN_SEEDS = [1, 2, 3, 4, 5]
SPLIT_SEEDS = {('[101, 102, 103]' if stage=='4' else '[]')}
CONFIG = dict(root=str(PROJECT_ROOT),output=str(OUTPUT_ROOT),stage="{stage}",
              budgets=LABEL_BUDGETS,seeds=TRAIN_SEEDS,split_seeds=SPLIT_SEEDS,
              epochs=80,cache="/content/kampa_feedback_audio_cache_v2_"+RUNTIME_ID)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = OUTPUT_ROOT / "launch_config.json"
CONFIG_PATH.write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2),encoding="utf-8")
(OUTPUT_ROOT/"launch_runtime.json").write_text(json.dumps(RUNTIME,ensure_ascii=False,indent=2),encoding="utf-8")
print("환경:",RUNTIME)\nprint("결과 저장:",OUTPUT_ROOT)
done = len(list((OUTPUT_ROOT/"runs").glob("*/complete.json")))
print(f"이 결과 폴더에 완료 표식이 있는 회차: {{done}}개. 해시를 확인한 뒤 건너뜁니다.")
for other in sorted(OUTPUT_ROOT.parent.parent.glob("*/{stage}")):
    count = len(list((other/"runs").glob("*/complete.json")))
    if other != OUTPUT_ROOT and count:
        print(f"다른 환경 폴더 {{other.parent.name}}에 완료 회차 {{count}}개가 있습니다. 런타임 환경이 달라 이번 실행에서는 이어 쓰지 않습니다.")
''')
        md('## 4. 학습 전 데이터·분할 검사')
        code('''subprocess.run([sys.executable, "-u", str(PACKAGE/"feedback_core.py"), "--config", str(CONFIG_PATH), "--prepare-only"], check=True)
import pandas as pd
display(pd.read_csv(OUTPUT_ROOT/"run_plan.csv"))
''')
        md('실제 입력 음원과 교사·점수 캐시를 검증합니다. 음원 누락이나 버전 불일치는 학습 전에 알려줍니다.')
        code('''audit_code = "import json,sys; import feedback_core as f; c=json.load(open(sys.argv[1])); f.audit_audio(c); s,_,_=f.load_inputs(c['root']); f.verified_teacher_scores(c['root'],s)"
subprocess.run([sys.executable,"-u","-c",audit_code,str(CONFIG_PATH)],cwd=PACKAGE,check=True)
''')
        md('합성 파형 두 개로 모델 출력을 확인합니다. 연구 자료 학습은 다음 단계에서 시작합니다.')
        code('''smoke_code = """import sys,torch
import feedback_core as f
torch.set_num_threads(2)
device=torch.device(f.runtime_metadata()['device'])
for kind in ['kampa','efficientat']:
    model=f.build_model(sys.argv[1],kind).to(device).eval()
    with torch.no_grad():
        result=model(torch.randn(2,48000,device=device)*.01)
    assert result.shape==(2,9) and torch.isfinite(result).all()
    print('모델 출력 확인:',kind,device,tuple(result.shape),flush=True)
    del model
"""
subprocess.run([sys.executable,"-u","-c",smoke_code,str(PROJECT_ROOT)],cwd=PACKAGE,check=True)
''')
        md('## 5. 실제 실험 실행\n완료된 회차는 11개 산출물의 해시를 확인한 뒤 건너뛰고, 중단된 회차는 마지막으로 완전히 저장한 에폭에서 재개합니다. 런타임이 끊기면 이 노트북을 다시 \'모두 실행\'하면 남은 회차부터 이어집니다. 회차마다 하는 중간 집계는 `KAMPA_결과집계.py`가 맡습니다.')
        code('''import datetime
log_path=OUTPUT_ROOT/("launch_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".log")
command=[sys.executable,"-u",str(AGGREGATOR),"--package",str(PACKAGE),"--config",str(CONFIG_PATH)]
with log_path.open("w",encoding="utf-8") as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in process.stdout:
            print(line,end="",flush=True);log.write(line);log.flush()
        returncode=process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
if returncode:
    raise RuntimeError(f"실행이 종료되었습니다(code={returncode}). 로그 확인: {log_path}")
print("계획 회차 실행 완료:",OUTPUT_ROOT)
''')
        md('## 6. 완료 결과 확인\n계획한 모든 회차가 끝나기 전의 평균·표준편차는 중간 결과입니다.')
        code('''subprocess.run([sys.executable,str(AGGREGATOR),"--package",str(PACKAGE),"--aggregate",str(OUTPUT_ROOT)],check=True)
plan=pd.read_csv(OUTPUT_ROOT/"run_plan.csv")
completed=list((OUTPUT_ROOT/"runs").glob("*/complete.json"))
print(f"완료 표식 {len(completed)}개 / 계획 {len(plan)}개 (집계 시 파일 해시 검증)")
summary=OUTPUT_ROOT/"results_summary.csv"
if summary.exists():display(pd.read_csv(summary))
else:print("아직 완료된 회차가 없습니다.")
''')
        nb=nbf.v4.new_notebook(cells=cells,metadata={'accelerator':'GPU','colab':{'name':name,'provenance':[],'gpuType':'L4'},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        for i,c in enumerate(nb.cells):c['id']=f'feedback-{stage}-{i:02d}'
        nbf.write(nb,BASE/'notebooks'/name)
    print('Bundled source SHA-256:',sha,'aggregator SHA-256:',agg_sha)
if __name__=='__main__':build()
