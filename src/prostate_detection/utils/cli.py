"""Shared helpers for thin CLI scaffolds."""

from __future__ import annotations

from argparse import Namespace


def summarize_args(args: Namespace) -> str:
    """Return a stable one-line summary of parsed CLI arguments."""
    parts = [f"{key}={value}" for key, value in sorted(vars(args).items())]
    return ", ".join(parts)


def not_implemented(script_name: str, args: Namespace) -> int:
    """Print a clear scaffold message and return a nonzero exit code."""
    print(f"{script_name} CLI scaffold")
    print(f"Arguments: {summarize_args(args)}")
    print("Status: not implemented yet. See docs/runbook.md and AGENTS.md.")
    return 2

