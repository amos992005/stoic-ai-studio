#!/usr/bin/env python3
"""
Local (CPU, no GPU, no torch) preview of the contact-sheet layout used by
`kaggle_kernel/character_forge.py`. Useful for iterating on the inspection grid
without burning Kaggle GPU minutes.

    python tools/preview_contact_sheet.py [--out preview_contact_sheet.png]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_kernel_module():
    """Import the kernel script without executing main()."""
    path = REPO_ROOT / "kaggle_kernel" / "character_forge.py"
    spec = importlib.util.spec_from_file_location("character_forge", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["character_forge"] = module
    spec.loader.exec_module(module)  # module-level code is config only
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="preview_contact_sheet.png")
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("pillow missing — pip install pillow", file=sys.stderr)
        return 1

    cf = load_kernel_module()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        master = tmpdir / cf.MASTER_FILENAME
        Image.new("RGB", cf.MASTER_SIZE, (40, 40, 46)).save(master)

        scene_paths = []
        for index, scene in enumerate(cf.SCENES):
            path = tmpdir / scene["filename"]
            img = Image.new("RGB", cf.SCENE_SIZE, (30 + index * 20, 34, 48))
            ImageDraw.Draw(img).text((40, 40), scene["title"], fill=(240, 240, 240))
            img.save(path)
            scene_paths.append(path)

        out = Path(args.out)
        timings: dict = {}
        cf.build_contact_sheet(
            master,
            scene_paths,
            ["01 · MASTER FACE", *[s["title"] for s in cf.SCENES]],
            out,
            timings,
        )
    print(f"preview written to {out} (timings={timings})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
