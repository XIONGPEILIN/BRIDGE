import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


class PrepareDatasetTest(unittest.TestCase):
    def test_verified_extraction_and_portable_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset';ext=dataset/'subject_condition'
            ext.mkdir(parents=True);targets=root/'targets';targets.mkdir()
            (targets/'target_1.png').write_bytes(b'target')
            assets=[];archive=ext/'assets.tar'
            names=['background.png','subject.png','sub.png','mask.png']
            with tarfile.open(archive,'w') as tf:
                for name in names:
                    relative='subject_condition/images/'+name;data=name.encode()
                    entry=tarfile.TarInfo(relative);entry.size=len(data);tf.addfile(entry,io.BytesIO(data))
                    assets.append({'path':relative,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            (ext/'assets_manifest.json').write_text(json.dumps(assets))
            (ext/'release_manifest.json').write_text(json.dumps({'archives':[{'path':'subject_condition/assets.tar','sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}]}))
            record={'image':'target_images/target_1.png',**{key:'subject_condition/images/'+name for key,name in zip(('edit_image','generated_subject_image','sub','back_mask'),names)}}
            for split in ('train','test'):(ext/f'{split}.json').write_text(json.dumps([record]))
            script=Path(__file__).resolve().parents[1]/'prepare_dataset.py'
            cmd=[sys.executable,str(script),'--dataset-root',str(dataset),'--target-images',str(targets),'--output-dir',str(root/'prepared'),'--extract']
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            row=json.loads((root/'prepared/train.json').read_text())[0]
            self.assertEqual(row['image'],str(targets/'target_1.png'))
            self.assertTrue(all(Path(value).is_file() for value in row.values()))
            (dataset/assets[0]['path']).write_bytes(b'changed')
            failed=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('Refusing to overwrite',failed.stderr)


if __name__=='__main__':unittest.main()
