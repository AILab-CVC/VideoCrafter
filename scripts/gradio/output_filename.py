"""Helpers for naming generated video artifacts."""

from uuid import uuid4


def make_video_filename(prompt: str) -> str:
    """Return a readable, per-request filename stem for a generated video."""
    prompt_slug = prompt.replace("/", "_slash_") if "/" in prompt else prompt
    prompt_slug = prompt_slug.replace(" ", "_") if " " in prompt_slug else prompt_slug
    return f"{prompt_slug[:30]}_{uuid4().hex}"
