"""Verify/extract the subject extension and materialize exact train/test paths."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--target-images',type=Path,required=True,
                   help='External original target_images directory containing target_N.png')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--extract',action='store_true')
    args=p.parse_args()
    root=args.dataset_root.resolve()
    extension=root/'subject_condition'
    release=json.loads((extension/'release_manifest.json').read_text())
    assets={entry['path']:entry for entry in json.loads((extension/'assets_manifest.json').read_text())}
    if args.extract:
        for archive in release['archives']:
            path=root/archive['path']
            if digest(path)!=archive['sha256']:
                raise ValueError(f'Archive checksum mismatch: {path}')
            with tarfile.open(path) as tf:
                for member in tf:
                    relative=Path(member.name)
                    if relative.is_absolute() or '..' in relative.parts or not member.isfile() or member.name not in assets:
                        raise ValueError(f'Unsafe or unlisted archive member: {member.name}')
                    target=root/relative
                    if not target.resolve().is_relative_to(root):
                        raise ValueError(f'Path escapes dataset root: {target}')
                    entry=assets[member.name]
                    if target.exists():
                        if digest(target)!=entry['sha256']:
                            raise ValueError(f'Refusing to overwrite different file: {target}')
                        continue
                    data=tf.extractfile(member).read()
                    if len(data)!=entry['size'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
                        raise ValueError(f'Member checksum mismatch: {member.name}')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as stream:stream.write(data)
            print('EXTRACTED',archive['path'],flush=True)
    prepared={}
    for split in ('train','test'):
        rows=json.loads((extension/f'{split}.json').read_text())
        for row in rows:
            for key in ('image','edit_image','generated_subject_image','sub','back_mask'):
                relative=Path(row[key])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError(row[key])
                path=args.target_images.resolve()/relative.name if key=='image' else root/relative
                if not path.is_file():raise FileNotFoundError(f'{key}: {path}')
                row[key]=str(path)
        prepared[split]=rows
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for split,rows in prepared.items():
        path=args.output_dir/f'{split}.json'
        with path.open('x') as stream:json.dump(rows,stream,ensure_ascii=False,indent=2)
        print('PREPARED',split,len(rows),path)


if __name__=='__main__':main()
