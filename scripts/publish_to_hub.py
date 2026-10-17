"""Publish a trained model to the Hugging Face Hub with a model card built from the real results.

    hf auth login                                          # once, with a token that has write access
    python scripts/publish_to_hub.py credit                # -> huggingface.co/<you>/nyaya-credit-risk
    python scripts/publish_to_hub.py sentiment   # -> huggingface.co/<you>/distilbert-financial-sentiment
    python scripts/publish_to_hub.py credit --dry-run      # only build dist/hub/credit/ to look at first

Options: --repo <user>/<name> to choose the name, --private to keep it private.
See docs/HF_HUB.md.
"""
from __future__ import annotations

import argparse
import sys

from compliance_ai.hub import HubError, publish


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a model to the Hugging Face Hub")
    parser.add_argument("model", choices=["credit", "sentiment"])
    parser.add_argument("--repo", help="<user>/<name> (default: <you>/nyaya-credit-risk or "
                                       "<you>/distilbert-financial-sentiment)")
    parser.add_argument("--private", action="store_true", help="create the repository as private")
    parser.add_argument("--dry-run", action="store_true", help="build the folder in dist/hub/ only")
    args = parser.parse_args()
    try:
        where = publish(args.model, args.repo, args.private, args.dry_run)
    except HubError as err:
        print(f"Not published: {err}")
        return 1
    print(f"Folder built: {where}" if args.dry_run else f"Published: {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
