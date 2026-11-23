#!/usr/bin/env python3
"""Download the lunar base source assets listed in manifest.yaml into raw/<category>/<slug>/.

Usage:
    python3 fetch_assets.py                # fetch everything not yet present
    python3 fetch_assets.py --only Z2-01 V-01
    python3 fetch_assets.py --attribution  # regenerate ATTRIBUTION.md only

Sketchfab models are taken from the login-free Objaverse mirror when available;
otherwise SKETCHFAB_API_TOKEN (sketchfab.com/settings/password) is required.
"""

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import zipfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OBJAVERSE = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "srb-lunar-base-assets"


def download(url, dest, headers=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with SESSION.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if dest.suffix == ".zip":
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
        dest.unlink()


def objaverse_paths(cache={}):
    if not cache:
        p = RAW / ".objaverse-object-paths.json.gz"
        if not p.exists():
            download(OBJAVERSE + "object-paths.json.gz", p)
        cache.update(json.load(gzip.open(p)))
    return cache


def fetch_sketchfab(f, out):
    uid = f["uid"]
    paths = objaverse_paths()
    if uid in paths:
        download(OBJAVERSE + paths[uid], out / f"{uid}.glb")
        return
    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        raise RuntimeError("not in Objaverse mirror; set SKETCHFAB_API_TOKEN")
    info = SESSION.get(
        f"https://api.sketchfab.com/v3/models/{uid}/download",
        headers={"Authorization": f"Token {token}"},
        timeout=60,
    )
    info.raise_for_status()
    info = info.json()
    fmt = "glb" if "glb" in info else "gltf"
    download(info[fmt]["url"], out / f"{uid}.{'glb' if fmt == 'glb' else 'zip'}")


def fetch_polyhaven(f, out):
    files = SESSION.get(f"https://api.polyhaven.com/files/{f['asset']}", timeout=60).json()
    g = files["gltf"][f["resolution"]]["gltf"]
    download(g["url"], out / Path(urllib.parse.urlparse(g["url"]).path).name)
    for rel, entry in g["include"].items():
        download(entry["url"], out / rel)


def fetch_url(f, out):
    for url in f["urls"]:
        name = urllib.parse.unquote(Path(urllib.parse.urlparse(url).path).name)
        if "ambientcg.com/get" in url:
            name = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["file"][0]
        download(url, out / name)


def fetch_fuel(f, out):
    owner, model = f["owner"], f["model"]
    q = urllib.parse.quote
    download(f"https://fuel.gazebosim.org/1.0/{q(owner)}/models/{q(model)}/tip/{q(model)}.zip", out / "model.zip")


def fetch_hf_dataset(f, out):
    tree = SESSION.get(
        f"https://huggingface.co/api/datasets/{f['repo']}/tree/main/{f['path']}?recursive=true", timeout=60
    ).json()
    for item in tree:
        if item["type"] != "file" or "/.thumbs/" in item["path"]:
            continue
        rel = item["path"][len(f["path"]) + 1 :]
        download(f"https://huggingface.co/datasets/{f['repo']}/resolve/main/{urllib.parse.quote(item['path'])}", out / rel)


def fetch_github_dir(f, out):
    url = f"https://codeload.github.com/{f['repo']}/tar.gz/{f['ref']}"
    with SESSION.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                parts = m.name.split("/", 1)
                if len(parts) == 2 and parts[1].startswith(f["path"] + "/") and m.isfile():
                    dest = out / parts[1][len(f["path"]) + 1 :]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as o:
                        shutil.copyfileobj(tar.extractfile(m), o)


def fetch_git_lfs(f, out):
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "clone", "--no-checkout", f"https://github.com/{f['repo']}.git", tmp], check=True, env=env)
        subprocess.run(["git", "-C", tmp, "checkout", f["ref"]], check=True, env=env)
        subprocess.run(["git", "-C", tmp, "lfs", "pull", "--include", ",".join(f["paths"])], check=True)
        for p in f["paths"]:
            src = Path(tmp) / p
            dest = out / Path(p).name
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)


FETCHERS = {
    "sketchfab": fetch_sketchfab,
    "polyhaven": fetch_polyhaven,
    "url": fetch_url,
    "fuel": fetch_fuel,
    "hf_dataset": fetch_hf_dataset,
    "github_dir": fetch_github_dir,
    "git_lfs": fetch_git_lfs,
}


def write_attribution(assets):
    lines = [
        "# Lunar base source assets — attribution",
        "",
        "Generated from `manifest.yaml`. Every asset below keeps its original license;",
        "modifications made for this project (rescaling, re-materialing, collision/joint authoring) are",
        "released under the same terms where the license requires it (CC-BY-SA).",
        "",
        "| ID | Asset | Author | License | Source |",
        "|---|---|---|---|---|",
    ]
    for a in assets:
        if a["fetch"]["type"] == "local":
            continue
        lines.append(f"| {a['id']} | {a['name']} | {a['author']} | {a['license']} | {a['page']} |")
    (ROOT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--attribution", action="store_true")
    args = ap.parse_args()
    assets = yaml.safe_load(open(ROOT / "manifest.yaml"))["assets"]
    write_attribution(assets)
    if args.attribution:
        return
    status = {}
    for a in assets:
        if args.only and a["id"] not in args.only:
            continue
        kind = a["fetch"]["type"]
        out = RAW / a["category"] / a["slug"]
        if kind in ("local", "manual"):
            status[a["id"]] = kind
            continue
        if out.exists() and any(out.iterdir()):
            status[a["id"]] = "present"
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            FETCHERS[kind](a["fetch"], out)
            status[a["id"]] = "ok"
        except Exception as e:  # report and continue with the rest of the manifest
            shutil.rmtree(out, ignore_errors=True)
            status[a["id"]] = f"FAILED: {e}"
        print(f"{a['id']:8s} {a['category']}/{a['slug']}: {status[a['id']]}", flush=True)
    (RAW / "fetch_status.json").write_text(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
