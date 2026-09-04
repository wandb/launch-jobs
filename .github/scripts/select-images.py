#!/usr/bin/env python3
"""Select container images affected by a set of changed repository paths."""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

MANIFEST_PATH = ".github/image-builds.json"


def is_within(path: str, directory: str) -> bool:
    path_parts = PurePosixPath(path).parts
    directory_parts = PurePosixPath(directory).parts
    return path_parts[: len(directory_parts)] == directory_parts


def load_images(manifest_path: Path) -> list[dict]:
    images = json.loads(manifest_path.read_text())["images"]
    names: set[str] = set()
    paths: set[str] = set()

    for image in images:
        name = image["image"]
        path = image["path"].rstrip("/")
        if name in names:
            raise ValueError(f"duplicate image name: {name}")
        if path in paths:
            raise ValueError(f"duplicate image path: {path}")
        names.add(name)
        paths.add(path)

        image["path"] = path
        image.setdefault("context", path)
        image.setdefault("dockerfile", f"{path}/Dockerfile.wandb")

        if not Path(image["context"]).is_dir():
            raise ValueError(f"missing build context for {name}: {image['context']}")
        if not Path(image["dockerfile"]).is_file():
            raise ValueError(f"missing Dockerfile for {name}: {image['dockerfile']}")

    return images


def affected(image: dict, changed_paths: list[str]) -> bool:
    for changed_path in changed_paths:
        if is_within(changed_path, image["path"]):
            return True

        shared_path = image.get("shared_path")
        if shared_path and is_within(changed_path, shared_path):
            excluded = image.get("shared_exclude_paths", [])
            if not any(is_within(changed_path, path) for path in excluded):
                return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=MANIFEST_PATH, type=Path)
    parser.add_argument(
        "--image",
        default=None,
        help="Force one image by name, or use 'all' to force every image",
    )
    args = parser.parse_args()

    images = load_images(args.manifest)
    changed_paths = [line.strip() for line in sys.stdin if line.strip()]

    if args.image:
        if args.image == "all":
            selected = images
        else:
            selected = [image for image in images if image["image"] == args.image]
            if not selected:
                raise ValueError(f"unknown image: {args.image}")
    elif MANIFEST_PATH in changed_paths:
        selected = images
    else:
        selected = [image for image in images if affected(image, changed_paths)]

    matrix = {
        "include": [
            {
                "image": image["image"],
                "context": image["context"],
                "dockerfile": image["dockerfile"],
            }
            for image in selected
        ]
    }
    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
