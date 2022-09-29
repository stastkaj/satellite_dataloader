from typing import Any, Dict, Hashable, List, Optional, Set, Tuple, Union
from enum import Enum
from pathlib import Path

from attrs import field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from cattrs.preconf.pyyaml import make_converter as make_yaml_converter
from satpy import Scene
from trollsift import Parser

from satdl.utils import tolist


yaml_converter = make_yaml_converter()
# tell cattrs how to (un-) structure Parser
yaml_converter.register_structure_hook(Parser, lambda data, _: Parser(data))


class SlotState(Enum):
    NotReady = 0  # some required files are missing
    Ready = 1  # all required files are ready, some optional are missing
    Complete = 2  # all required and optional files are missing


@frozen
class SlotFiles:
    """List of files belonging to a single slot that can create SatpyScene."""

    reader: str
    required_files: List[Optional[Path]]
    optional_files: List[Optional[Path]]
    attrs: Dict[str, Hashable]
    _key: Hashable

    @property
    def key(self) -> Hashable:
        return self._key or tuple(self.attrs.items())

    @classmethod
    def empty_slot(
        cls,
        reader: str,
        attrs: Dict[str, Any],
        n_required: int,
        n_optional: int,
        key: Optional[Hashable] = None,
    ) -> "SlotFiles":
        return cls(
            reader=reader,
            required_files=[None] * n_required,
            optional_files=[None] * n_optional,
            attrs=attrs,
            key=key,
        )

    @property
    def filenames(self) -> List[Path]:
        return [f for f in self.required_files + self.optional_files if f is not None]

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


@frozen
class SatpySlotFiles(SlotFiles):
    @property
    def scene(self) -> Scene:
        return Scene(reader=self.reader, filenames=[str(f) for f in self.filenames])


@frozen
class SlotDefinition:  # type:ignore
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
    required_file_masks: List[Parser] = field(converter=tolist, validator=deep_iterable(instance_of(Parser)))
    optional_file_masks: Optional[List[Parser]] = field(
        converter=tolist, validator=optional(deep_iterable(instance_of(Parser)))  # type: ignore
    )
    ignored_attrs: Set[str] = field(  # type: ignore
        converter=lambda x: set(tolist(x)), validator=optional(deep_iterable(instance_of(str)))  # type: ignore
    )

    @classmethod
    def from_yaml_file(cls, path: Union[str, Path]) -> "SlotDefinition":
        """Construct class from definition in yaml file"""
        with open(path, "rt", encoding="utf-8") as f:
            data_as_str = f.read()

        return yaml_converter.loads(data_as_str, cls)

    def new_slot(self, attrs: Dict[str, Any], key: Optional[Hashable] = None) -> SlotFiles:
        return SatpySlotFiles.empty_slot(
            reader=self.reader,
            attrs=attrs,
            n_required=len(self.required_file_masks),
            n_optional=len(self.optional_file_masks) if self.optional_file_masks else 0,
            key=key,
        )

    def attrs2slot_attrs(self, attrs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        slot_attrs = {k: v for k, v in attrs.items() if k not in self.ignored_attrs}.items()
        return tuple(sorted(slot_attrs))


@frozen
class SegmentGatherer:
    slot_definition: SlotDefinition

    def gather(self, path: Union[str, Path]) -> Dict[Tuple[Tuple[str, Any], ...], SatpySlotFiles]:
        path = Path(path)

        # find all files
        all_files = [f for f in path.rglob("*") if f.is_file()]

        # find if a file is required or optional and group them to slots according to their attributes
        slots = {}
        for file in all_files:
            for masks, file_list_name in (
                (self.slot_definition.required_file_masks, "required_files"),
                (self.slot_definition.optional_file_masks or [], "optional_files"),
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
                        slots[slot_key] = self.slot_definition.new_slot(dict(slot_key))

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
        return {attrs: slot for attrs, slot in slots.items() if slot.is_ready}  # type: ignore
