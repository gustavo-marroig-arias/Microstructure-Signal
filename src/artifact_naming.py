from __future__ import annotations

import re


_TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def artifact_suffix(artifact_tag: str | None) -> str:
    if artifact_tag is None:
        return ""

    tag = artifact_tag.strip()
    if not tag:
        return ""

    if not _TAG_RE.fullmatch(tag):
        raise ValueError(
            "artifact_tag may only contain letters, numbers, underscores, dots, "
            "and hyphens."
        )

    return f"_{tag}"


def tagged_artifact_stem(
    base: str,
    symbol: str,
    start: str,
    end: str,
    artifact_tag: str | None = None,
) -> str:
    return f"{base}{artifact_suffix(artifact_tag)}_{symbol}_{start}_to_{end}"


def frozen_model_artifact_stem(
    model_name: str,
    symbol: str,
    start: str,
    end: str,
    horizon: int,
    artifact_tag: str | None = None,
) -> str:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    base = tagged_artifact_stem(
        model_name,
        symbol,
        start,
        end,
        artifact_tag,
    )
    return f"{base}_h{horizon}"
