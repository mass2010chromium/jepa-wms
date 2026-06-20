#!/usr/bin/env python3
"""
Inspect the `heads` entry of a JEPA-WM checkpoint.

Downloads a checkpoint from the facebook/jepa-wms Hugging Face repo and prints
the top-level keys plus a structured view of whatever lives under `heads`, so
you can see whether a state head was actually bundled with a released model.

Requires:  pip install torch huggingface_hub

Usage:
    python inspect_heads.py                         # defaults to jepa_wm_metaworld
    python inspect_heads.py jepa_wm_droid.pth.tar   # any checkpoint filename
"""

import sys
import torch

REPO_ID = "facebook/jepa-wms"
DEFAULT_FILE = "jepa_wm_metaworld.pth.tar"  # smallest checkpoint, quick to pull


def describe(obj, indent=2, max_items=None):
    """Recursively print a compact view of nested dicts / tensors."""
    pad = " " * indent
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            if max_items is not None and i >= max_items:
                print(f"{pad}... ({len(obj) - max_items} more keys)")
                break
            if isinstance(v, torch.Tensor):
                print(f"{pad}{k}: Tensor{tuple(v.shape)} {v.dtype}")
            elif isinstance(v, dict):
                print(f"{pad}{k}: dict ({len(v)} keys)")
                describe(v, indent + 4, max_items=max_items)
            else:
                print(f"{pad}{k}: {type(v).__name__}")
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}Tensor{tuple(obj.shape)} {obj.dtype}")
    else:
        print(f"{pad}{type(obj).__name__}: {obj!r}")


def main():
    path = sys.argv[1]

    ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print(f"Checkpoint is a {type(ckpt).__name__}, not a dict — stopping.")
        return

    print("Top-level checkpoint keys:")
    for k, v in ckpt.items():
        extra = ""
        if isinstance(v, dict):
            extra = f" ({len(v)} keys)"
        elif isinstance(v, torch.Tensor):
            extra = f" Tensor{tuple(v.shape)}"
        print(f"  - {k}: {type(v).__name__}{extra}")

    heads = ckpt.get("model", None)
    print("\n--- 'model' contents ---")
    if heads is None:
        print("No 'model' entry (or it is None) — no bundled model in this checkpoint.")
    elif isinstance(heads, dict) and len(heads) == 0:
        print("'model' is an empty dict — no head was trained/saved here.")
    else:
        # Show head names and, one level down, their parameter tensors.
        describe(heads, indent=2)
        print(
            "\nTip: parameter names like '...state_head...' or '...decoder...' "
            "tell you which kind of head (state vs image) is present."
        )


if __name__ == "__main__":
    main()
