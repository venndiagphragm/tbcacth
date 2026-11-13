"""Fetch and verify only the pinned BEATs encoder, never study data."""
from pathlib import Path
import argparse
import json
import sys
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_v5_assets import fetch

ASSETS = ROOT / 'assets_v5.json'
REPO_ID = 'mooneyko/BEATs'
REVISION = '18cfdf9d43820b2db86c6dfde4ae2f7531d1f5ad'
FILENAME = 'BEATs_iter3_plus_AS2M.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='verify a local weight file without downloading')
    args = parser.parse_args()
    record = json.loads(ASSETS.read_text(encoding='utf-8'))
    asset = next(item for item in record['assets']
                 if item['path'] == f'pretrained/v5/{FILENAME}')
    target = ROOT / asset['path']
    if not target.is_file():
        if args.check_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME,
                                          revision=REVISION, local_dir=str(target.parent)))
        if downloaded.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected local download path: {downloaded}')
        try:
            checked = fetch(asset['url'], target, asset['sha256'])
            if checked['bytes'] != asset['bytes']:
                raise ValueError('Downloaded BEATs asset has unexpected size')
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    else:
        checked = fetch(asset['url'], target, asset['sha256'])
        if checked['bytes'] != asset['bytes']:
            raise ValueError(f'Pinned BEATs size mismatch: {target}')
    print(f"BEATs verified: {target} ({asset['bytes']} bytes, SHA-256 {asset['sha256']})")


if __name__ == '__main__':
    main()
