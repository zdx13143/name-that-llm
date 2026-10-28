"""Package the existing Linux runtime with only the six R10 inference dependencies."""
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile

from inference_round10 import VERSIONS, digest, verify_assets, write_json


def main():
    if sys.platform != "linux":
        raise RuntimeError("Package in Linux to preserve binary modes and symlinks")
    module = Path(__file__).resolve().parents[1]
    runtime = module / "outputs/kaggle_r6_runtime/python/cpython-3.11.17-linux-x86_64-gnu"
    if Path(sys.prefix).resolve() != runtime.resolve():
        raise ValueError("Use the existing fixed Python 3.11 runtime")
    assets = module / "outputs/kaggle_r10_assets/ntl-r10-assets"
    manifest_sha = digest(assets / "manifest.json")
    verify_assets(assets, manifest_sha)
    allowed = set()
    for name, version in VERSIONS.items():
        dist = importlib.metadata.distribution(name)
        if dist.version != version:
            raise ValueError(f"Runtime version changed: {name}")
        if dist.files is None:
            raise ValueError(f"Missing installed file inventory: {name}")
        allowed.update(Path(str(path)).parts[0] for path in dist.files if Path(str(path)).parts[0] != "..")
    output = module / "outputs/kaggle_r10_package"
    output.mkdir(exist_ok=False)
    def archive_filter(info):
        parts = Path(info.name).parts
        if "__pycache__" in parts:
            return None
        if "site-packages" in parts:
            index = parts.index("site-packages") + 1
            if len(parts) > index and parts[index] not in allowed:
                return None
        if info.issym() and Path(info.linkname).is_absolute():
            raise ValueError(f"Absolute symlink rejected: {info.name}")
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info
    payload = output / "ntl_r10_payload.tgz.bin"
    with tarfile.open(payload, "x:gz", compresslevel=3) as archive:
        archive.add(assets, arcname=assets.name, filter=archive_filter)
        archive.add(runtime, arcname=runtime.name, filter=archive_filter)
    receipt = {"state": "packaged_not_yet_validated", "payload_file": payload.name, "bytes": payload.stat().st_size,
               "payload_sha256": digest(payload), "manifest_sha256": manifest_sha,
               "runtime_versions": VERSIONS, "included_site_package_roots": sorted(allowed)}
    write_json(output / "package_receipt.json", receipt, True)
    (output / "README.txt").write_text("Private R10 inference bundle: frozen weighted SVC, source attribution, artificial compatibility texts and isolated Linux Python 3.11.\nNo competition raw data, protected holdout, OOF or credentials. The .bin is a gzip tar read by the companion Notebook.\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
