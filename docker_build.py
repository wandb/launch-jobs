#!/usr/bin/env python3

import os
import subprocess
import datetime
from typing import Dict, List, Optional


def exec_read(cmd: str) -> str:
    try:
        proc = subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise e

    return proc.stdout.decode("utf-8").rstrip()


def exec_stream(cmd: str) -> str:
    return subprocess.run(
        cmd.replace("\n", "").replace("\\", ""), shell=True, check=True
    )


def print_green(msg):
    print("\033[92m\033[1m{}\033[00m\n".format(msg))


DIRTY = "-dirty" if exec_read("git status -s") != "" else ""

GIT_SHA = exec_read("git describe --match= --always --abbrev=40")
GIT_SHA_DIRTY = GIT_SHA + DIRTY  # will be $GIT_SHA if repository is not dirty
GIT_BRANCH = exec_read("git rev-parse --abbrev-ref HEAD").replace("/", "-")
GIT_BRANCH_DIRTY = GIT_BRANCH + DIRTY  # will be $GIT_BRANCH if repository is not dirty
GIT_PARENT_SHA = exec_read("git describe --match= --always --abbrev=40 HEAD^")

REGISTRY = ""
NO_CACHE = bool(os.getenv("NO_CACHE", False))
CI = bool(os.getenv("CI", False))
BUILD_DATE = exec_read("date +%Y-%m-%d")

# Target Tags:
#
# WIP commits will get :SHA-dirty and :branch-dirty, whereas builds from clean
# commits will get :SHA and :branch
WRITE_TAGS = [GIT_SHA_DIRTY, GIT_BRANCH_DIRTY]


def qualified_image_name(image: str, tag: str) -> str:
    return os.path.join(REGISTRY, f"{image}:{tag}")


def _build_common(
    image: str,
    context_path: str,
    dockerfile: str,
    platforms: List[str] = ["linux/amd64", "linux/arm64"],
) -> str:
    command = f"docker buildx build {context_path}"
    command += f" \ \n  --file={dockerfile}"

    for platform in platforms:
        command += f" \ \n  --platform={platform}"

    # This line tells BuildKit to include metadata so that all layers of the
    # result image can be used as a cache source when pulled from the registry.
    # Note that this DOES NOT HAPPEN BY DEFAULT: without this flag, the
    # intermediate layers become part of your LOCAL cache, but can't be used
    # as cache from the registry.
    command += " \ \n  --cache-to=type=inline"

    isodate = datetime.datetime.utcnow().isoformat()
    command += f" \ \n  --label=ai.wandb.built-at={isodate}"
    command += f" \ \n  --label=ai.wandb.built-for-branch={GIT_BRANCH}"

    # these options manipulate the output type -- refer to
    # https://docs.docker.com/engine/reference/commandline/buildx_build/#options
    # and https://docs.docker.com/engine/reference/commandline/buildx_build/#output for
    # details
    if CI:
        # Everyone can read from the registry, but only authorized users can push.
        print_green(
            f'Image will be pushed to {REGISTRY}. Set CI="" to load into the local docker daemon.'
        )
        command += " \ \n  --push"
    else:
        # if we're not pushing, we'll assume we want to load the image into the running
        # docker agent (there's really no point to building if you don't do at least one of these two
        # things, since the resultant image would just remain in cache, unusable)
        print_green(
            f"Image will be loaded into the local docker daemon. Set CI=1 to push to {REGISTRY}"
        )
        command += " \ \n  --load"

    return command


CACHE_FROM_TAGS = [
    "latest-deps",
    GIT_SHA_DIRTY,
    GIT_SHA,
    GIT_PARENT_SHA,
    GIT_BRANCH,
    "master",
]


def build(
    image: str,
    context_path: str,
    dockerfile: str,
    cache_from_image: Optional[str] = None,
    extra_write_tags: List[str] = [],
    build_args: Dict[str, str] = {},
    build_contexts: Dict[str, str] = {},
    target: Optional[str] = None,
    platforms: List[str] = ["linux/arm64", "linux/amd64"],
) -> str:
    """
    Builds an image, using previous builds as cache.

    Recent builds of `cache_from_image` will be searched for matching cache layers.
    In most cases, `cache_from_image` will be the same as `image`, but for
    complex multi-stage builds, it's sometimes useful to use an earlier stage
    as the cache source instead.

    Args:
        image: The name of the image to build.
        context_path: The path to the directory to use as the build context.
        dockerfile: The path to the Dockerfile.
        cache_from_image: The name of the image to use as the cache source.
        extra_write_tags: A list of additional tags to apply to the image (in additiion
                    to the standard SHA and branch name tags).
    """

    cache_from_image = cache_from_image if cache_from_image else image

    # Cache Sources:
    #
    # The goal here is to make sure that every build uses a cache from the
    # closest possible successfully-built commit. So:
    #
    # - all builds will try to reference the latest cacheless build (to ensure deps
    #   are up to date)
    # - a rebuild on the same commit will reuse the original build
    # - a build on a WIP commit will start from the clean commit it started from
    # - a build on a newly-committed commit will start from its parent, IF its
    #   parent built successfully
    # - a build on a newly-committed commit will start from the most-recent
    #   successful build on its branch
    # - if all else fails, a build will start from the latest successful master build

    print_green(f"Building {image} at commit {GIT_SHA_DIRTY}...")

    command = _build_common(image, context_path, dockerfile, platforms)

    if target:
        command += f" \ \n  --target={target}"

    for arg, value in build_args.items():
        command += f" \ \n  --build-arg {arg}={value}"

    for name, context in build_contexts.items():
        command += f" \ \n  --build-context {name}={context}"

    for tag in CACHE_FROM_TAGS:
        full_name = qualified_image_name(cache_from_image, tag)
        command += f" \ \n  --cache-from={full_name}"

    for tag in WRITE_TAGS + extra_write_tags:
        full_name = qualified_image_name(image, tag)
        command += f" \ \n  --tag={full_name}"

    if not CI:
        # development tooling sometimes looks for service:latest, so we should
        # make sure that we tag newly-built images that way when building for
        # local development
        print_green("Because CI==False, image will also be tagged `latest`")
        full_name = qualified_image_name(image, "latest")
        command += f" \ \n  --tag={full_name}"

    print(command)
    print("")

    exec_stream(command)


def _get_image_paths_dir_and_subdir(repo: str):
    """Add top level and sub-directory paths that contain dockerfiles"""
    image_paths = []
    for dir in os.listdir(os.path.join(repo, "jobs")):
        if not os.path.isdir(os.path.join(repo, "jobs", dir)):
            continue

        # if top level dir has Dockerfile, thats the image dir
        if os.path.exists(os.path.join(repo, "jobs", dir, "Dockerfile")):
            image_paths += [dir]
            continue

        # go through subdirs for folders that container Dockerfile
        for subdir in os.listdir(os.path.join(repo, "jobs", dir)):
            if os.path.isdir(os.path.join(repo, "jobs", dir, subdir)):
                if os.path.exists(
                    os.path.join(repo, "jobs", dir, subdir, "Dockerfile")
                ):
                    image_paths += [os.path.join(dir, subdir)]
    return image_paths


if __name__ == "__main__":
    repo = os.path.dirname(os.path.realpath(__file__))
    image_paths = _get_image_paths_dir_and_subdir(repo)

    for image in image_paths:
        if "sweep_controllers" not in image:
            continue

        extra_write_tags = []

        if GIT_BRANCH_DIRTY == "master":
            extra_write_tags.append("latest")

        image_formatted = image.replace("/", "_")

        build(
            image=f"wandb/job_{image_formatted}",
            context_path=os.path.join(repo, "jobs", image),
            dockerfile=os.path.join(repo, "jobs", image, "Dockerfile"),
            cache_from_image=f"wandb/{image_formatted}",
            extra_write_tags=extra_write_tags,
        )
