"""Shared helpers for the LoTA-QAF reproduction."""

import os


def resolve_pretrained(name: str) -> str:
    """Turn a hub id into the local snapshot dir when the cache already has it.

    GPTQModel.load() calls huggingface_hub.list_repo_files() for anything that is not a local
    directory, which fails on a compute node with no route out even though every file is
    already in HF_HOME. Resolving the id here keeps the offline path purely local; an id that
    genuinely is not cached is passed through unchanged so the normal download still works.
    """
    if os.path.isdir(name):
        return name
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(name, local_files_only=True)
        print(f"[lota] {name} -> {local}")
        return local
    except Exception as exc:  # noqa: BLE001
        print(f"[lota] {name} not resolvable from the local cache ({exc.__class__.__name__}); "
              f"using the id as given")
        return name
