"""Resume, verify, and extract the official FMA-small research dataset."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


ARCHIVES = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
}
SOURCE = "https://os.unil.cloud.switch.ch/fma/"


def checksum(path):
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/audio"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An audio download is already running in this directory")
        marker = args.output / "fma_download_complete.json"
        if marker.exists() and json.loads(marker.read_text()).get("sha1") == ARCHIVES:
            print(f"Verified dataset already extracted in {args.output}", flush=True)
            return
        if shutil.disk_usage(args.output).free < 20 * 1024**3:
            raise RuntimeError("Need at least 20 GiB free for archives and extracted FMA-small")
        for name, expected in ARCHIVES.items():
            archive = args.output / name
            if not archive.exists():
                partial = args.output / (name + ".part")
                print(f"Downloading {SOURCE + name} -> {partial}", flush=True)
                subprocess.run(["curl", "--fail", "--location", "--continue-at", "-",
                                "--retry", "5", "--retry-delay", "5", "--connect-timeout", "20",
                                "--speed-limit", "1024", "--speed-time", "120",
                                "--output", str(partial), SOURCE + name], check=True)
                if checksum(partial) != expected:
                    raise RuntimeError(f"Checksum mismatch for {partial}; archive was not extracted")
                partial.rename(archive)
            elif checksum(archive) != expected:
                raise RuntimeError(f"Checksum mismatch for existing {archive}; no files overwritten")
            print(f"Verified SHA-1 for {name}; extracting", flush=True)
            destination = args.output.resolve()
            with zipfile.ZipFile(archive) as contents:
                for entry in contents.infolist():
                    target = (destination / entry.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise RuntimeError(f"Unsafe archive path: {entry.filename}")
                    contents.extract(entry, destination)
        marker.write_text(json.dumps({"source": SOURCE, "sha1": ARCHIVES,
                                      "audio_root": str(args.output / "fma_small"),
                                      "metadata": str(args.output / "fma_metadata" / "tracks.csv")}, indent=2) + "\n")
        print(f"FMA-small ready: {args.output / 'fma_small'}", flush=True)
        print("Download only: audio codec preparation/training has not been started.", flush=True)


if __name__ == "__main__":
    main()
