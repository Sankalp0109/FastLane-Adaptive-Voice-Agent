from dataclasses import fields

from fastlane.config import FASTLANE, SYSTEMS


def test_twelve_systems():
    assert len(SYSTEMS) == 12


def test_each_variant_changes_one_part_of_fastlane():
    # A3 removes checks *and* moves rules to the prompt; A1 replaces tuning with a prompt.
    # Both are one conceptual change but touch two flags, so allow up to two.
    ignored = {"name", "description", "tags"}
    for name, cfg in SYSTEMS.items():
        if not name.startswith("A"):
            continue
        changed = [f.name for f in fields(cfg) if f.name not in ignored and getattr(cfg, f.name) != getattr(FASTLANE, f.name)]
        assert 1 <= len(changed) <= 2, (name, changed)
