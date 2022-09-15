from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
from functools import partial
from pathlib import Path

from attrs import define, field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from satpy import Scene
from trollsift import Parser

from satdl.utils import tolist


class SlotState(Enum):
    NotReady = 0  # some required files are missing
    Ready = 1     # all required files are ready, some optional are missing
    Complete = 2  # all required and optional files are missing

@frozen
class SlotFiles:
    """List of files belonging to a single slot that can create SatpyScene."""
    reader: str
    required_files: List[Path]
    optional_files: List[Path]
    attrs: Dict[str, Any]

    @classmethod
    def empty_slot(cls, reader: str, attrs: Dict[str, Any], n_required: int, n_optional: int):
        return cls(reader=reader, required_files=[None] * n_required, optional_files=[None] * n_optional, attrs=attrs)

    @property
    def filenames(self) -> List[Path]:
        return self.required_files + self.optional_files

    @property
    def state(self) -> SlotState:
        if any(f is None for f in self.required_files):
            # some required files are missing
            return SlotState.NotReady

        if any(f is None for f in self.optional_files):
            # all required files ready but some optional are missing
            return SlotState.Ready

        # all required and optional files are ready
        return SlotState.Complete

    @property
    def is_ready(self) -> bool:
        return self.state.value > 0

    @property
    def scene(self) -> Scene:
        return Scene(reader=self.reader, filenames=self.filename)

@frozen
class SlotDefinition:
    """List of file masks belonging to a single slot with specification required/optional.

    Slot = set of files with non-conflicting values of their attributes.

    Attrs:
        reader: Satpy reader that can read these files
        required_file_masks: List of masks of files required for the slot to be complete
        optional_file_masks: List of masks of files that belong to the slot but are optional
        ignored_attrs: Values of these attributes are not considered when deciding whether
            two files belong to the same slot
    """
    reader: str
    required_file_masks: List[Parser] = field(
        converter=partial(tolist, converter=Parser),
        validate=deep_iterable(instance_of(Parser))
    )
    optional_file_masks: Optional[List[Parser]] = field(
        converter=partial(tolist, converter=Parser),
        validate=optional(deep_iterable(instance_of(Parser)))
    )
    ignored_attrs: Optional[Set[str]] = field(
        converter=lambda x: set(tolist(x)),
        validate=optional(deep_iterable(instance_of(str)))
    )

    def new_slot(self, attrs: Dict[str, Any]) -> SlotFiles:
        return SlotFiles.empty_slot(
            reader=self.reader,
            attrs=attrs,
            n_required=len(self.required_file_masks),
            n_optional=len(self.optional_file_masks)
        )

    def attrs2slot_attrs(self, attrs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        slot_attrs = {k: v for k, v in attrs.items() if k not in self.ignored_attrs}.items()
        return tuple(sorted(slot_attrs))


@frozen
class SegmentGatherer:
    slot_definition: SlotDefinition

    def gather(self, path: Path) -> Dict[Tuple[Tuple[str, Any], ...], SlotFiles]:
        # find all files
        all_files = [f for f in path.rglob('*') if f.is_file()]

        slots = {}
        # find if file is required/optional and group to slots according to their attributes
        for file in all_files:
            for masks, file_list_name in (
                (self.slot_definition.required_file_masks, 'required_files'),
                (self.slot_definition.optional_file_masks, 'optional_files')
            ):
                found = False
                for imask, mask in enumerate(masks):
                    try:
                        attrs = mask.parse(file.name)
                    except ValueError:
                        continue

                    # find slot key
                    slot_key = self.slot_definition.attrs2slot_attrs(attrs)
                    if slot_key not in slots:
                        # create empty slot
                        slots[slot_key] = self.slot_definition.new_slot(attrs)

                    # get slot's filename list
                    slot_file_list = getattr(slots[slot_key], file_list_name)
                    if slot_file_list[imask] is not None:
                        # found duplicate file
                        # TODO: raise or warn
                        continue

                    # save filename and break
                    slot_file_list[imask] = file
                    found = True
                    break

                if found:
                    break

        # return only those slots that are ready
        return {attrs: slot for attrs, slot in slots.items() if slot.is_ready}
