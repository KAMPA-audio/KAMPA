"""Verify reviewed source, the notebook code ZIP and aggregator SHA-256 pins and release file hashes."""
from pathlib import Path
import argparse,ast,hashlib,io,json,zipfile
import nbformat
BASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():return sorted(p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.suffix not in ('.pyc','.pyo') and p.name!='release_manifest.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    source={p.relative_to(BASE/'src').as_posix():p.read_bytes() for p in files() if (BASE/'src') in p.parents}
    expected=(BASE/'notebooks/KAMPA_추가실험_실행코드.zip').read_bytes();expected_sha=hashlib.sha256(expected).hexdigest()
    with zipfile.ZipFile(io.BytesIO(expected)) as archive:
        assert set(archive.namelist())==set(source),'ZIP inventory differs from src'
        for name,data in source.items():assert archive.read(name)==data,name
    aggregator_sha=sha(BASE/'notebooks/KAMPA_결과집계.py')
    notebooks=sorted((BASE/'notebooks').glob('*.ipynb'))
    for p in notebooks:
        n=nbformat.read(p,as_version=4);nbformat.validate(n);pinned=pinned_aggregator=None
        assert n.metadata.get('colab',{}).get('gpuType')=='L4',p
        for cell in n.cells:
            if cell.cell_type!='code':continue
            assert not cell.outputs and cell.execution_count is None
            assert 'PAYLOAD' not in cell.source and 'b64decode' not in cell.source,p
            assert 'from feedback_core import aggregate' not in cell.source and '"feedback_core.py"),"--config"' not in cell.source,p
            tree=ast.parse(cell.source)
            for node in tree.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_SHA256' for t in node.targets):pinned=ast.literal_eval(node.value)
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AGGREGATOR_SHA256' for t in node.targets):pinned_aggregator=ast.literal_eval(node.value)
        assert pinned==expected_sha,p
        assert pinned_aggregator==aggregator_sha,p
    for p in files():
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    manifest=BASE/'release_manifest.json'
    current={p.relative_to(BASE).as_posix():sha(p) for p in files()}
    if args.refresh:manifest.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    expected_hashes=json.loads(manifest.read_text());assert expected_hashes==current,'Release inventory/hash mismatch'
    print(f'Verified {len(current)} package files, {len(notebooks)} notebooks pinned to ZIP {expected_sha[:12]} and aggregator {aggregator_sha[:12]}, {len(source)} bundled source files, Python syntax.')
if __name__=='__main__':main()
