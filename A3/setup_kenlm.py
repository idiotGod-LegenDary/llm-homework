"""Install the pinned Windows KenLM tools with SHA256 verification."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
NAMES={"lmplz.exe", "build_binary.exe", "query.exe"}

def install(archive, destination, expected_hash):
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!=expected_hash:
        raise ValueError(f"Archive SHA256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as z:
        selected={}
        for member in z.infolist():
            name=member.filename.replace("\\", "/").rsplit("/",1)[-1]
            if not member.is_dir() and name in NAMES:
                if name in selected:raise ValueError(f"Duplicate executable: {name}")
                if member.file_size>32*1024*1024:raise ValueError("Executable too large")
                selected[name]=member
        if set(selected)!=NAMES:raise ValueError("Archive lacks required KenLM tools")
        destination.mkdir(parents=True,exist_ok=True)
        for name,member in selected.items():
            # Use fixed basenames: no archive path is extracted onto the filesystem.
            (destination/name).write_bytes(z.read(member))
    print(f"Installed {len(NAMES)} verified tools in {destination}")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,help="Use a local zip instead of downloading")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"tools"/"bin")
    args=parser.parse_args()
    if os.name!="nt":parser.error("This pinned package is for Windows; build upstream KenLM on other platforms")
    manifest=json.loads((ROOT/"tools"/"tool_manifest.json").read_text(encoding="utf8"))
    if args.archive:
        install(args.archive,args.output_dir,manifest["sha256"])
    else:
        with tempfile.TemporaryDirectory(prefix="kenlm-") as temporary:
            archive=Path(temporary)/"kenlm.zip"
            urllib.request.urlretrieve(manifest["asset_url"],archive)
            install(archive,args.output_dir,manifest["sha256"])

if __name__=="__main__":main()
