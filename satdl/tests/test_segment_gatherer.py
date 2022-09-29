from collections import abc
import os
from pathlib import Path

import importlib_resources as resources
import pytest

from satdl.datasets._segment_gatherer import SegmentGatherer, SlotDefinition


@pytest.fixture
def slot_definition() -> SlotDefinition:
    with resources.as_file(resources.files("satdl") / "definitions" / "slot_MSG_CE.yaml") as path:
        yield SlotDefinition.from_yaml_file(path)


@pytest.fixture
def hrit_path() -> Path:
    path = os.environ.get("SATDL_TEST_HRIT_PATH")
    if not path:
        pytest.skip("$SATDL_TEST_HRIT_PATH not set")

    return Path(path)


def test_slot_definition_from_yaml(slot_definition: SlotDefinition) -> None:
    # HRV + 11 channels + EPI + PRO
    assert len(slot_definition.required_file_masks) == 9 + 11 * 3 + 2
    assert len(slot_definition.optional_file_masks) == 11 * 5
    assert slot_definition.reader == "seviri_l1b_hrit"
    assert len(slot_definition.ignored_attrs) == 0
    assert all(
        isinstance(mask.fmt, str)
        for mask in slot_definition.optional_file_masks + slot_definition.required_file_masks
    )


def test_segment_gatherer(hrit_path: Path, slot_definition: SlotDefinition) -> None:
    segment_gatherer = SegmentGatherer(slot_definition)

    slots = segment_gatherer.gather(hrit_path)

    # at least one slot was found
    assert slots

    # can construct satpy scene
    for slot in slots.values():
        slot.scene

    # keys contain datatetime and channel
    for slot_attrs in slots:
        for expected_attr in ["datetime", "platform_shortname"]:
            assert expected_attr in [attr_tuple[0] for attr_tuple in slot_attrs]

    # slot key is stored and hashable
    for slot in slots.values():
        assert slot.key is not None
        assert isinstance(slot.key, abc.Hashable)
