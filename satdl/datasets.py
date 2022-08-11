from __future__ import annotations

from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generator,
    ItemsView,
    Iterable,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from collections import defaultdict
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
from trollsift import Parser
import xarray as xr

from satdl.utils import image2xr, tolist


_logger = logging.getLogger(__name__)


def _get_get_image(
    georef: Optional[Union[str, Path, xr.DataArray]]
) -> Callable[[Union[str, Path]], xr.DataArray]:
    def _get_image(path: Union[str, Path]) -> xr.DataArray:
        _logger.debug(f"loading georeferenced image {path}")
        return image2xr(path, georef=georef).load()

    return _get_image


class StaticImageFolderDataset(Mapping[str, xr.DataArray]):
    def __init__(
        self,
        base_folder: Union[str, Path],
        file_mask: Union[str, Parser],
        georef: Optional[Union[str, Path, xr.DataArray]] = None,
        max_cache: Optional[int] = 0,
    ) -> None:
        """Create ImageFolderDataset

        Note: content of the folder is scanned only once, at the class creation

        Parameters
        ----------
        base_folder : str or Path
            root folder of the data
        file_mask : str or trollsift.Parser
            mask of image names specifying attributes in the file name. Must not contain wildcards '*' or '?',
            should be relatie to base_folder
        georef : str or Path or xr.DataArray or None
            external georeference for plain images, optional
        max_cache: int, optional
            Maximum number of images that will be cached.
        """
        self._base_folder = Path(base_folder)
        if not self._base_folder.exists():
            raise ValueError(f"base folder {base_folder} does not exist")
        self._file_mask = Parser(file_mask)
        self._georef = georef  # TODO: validate georeference
        self._files = list(self._base_folder.rglob(self._file_mask.globify()))
        self._attrs = {self._filename2key(f): self._extract_attrs(f, relative=False) for f in self._files}

        self._get_image = lru_cache(max_cache)(_get_get_image(self._georef))

    def __len__(self) -> int:
        return len(self._files)

    def _extract_attrs(self, filename: Union[str, Path], relative: bool = False) -> Dict[str, Any]:
        key = str(filename) if relative else self._filename2key(filename)
        return self._file_mask.parse(key)

    def _filename2key(self, filename: Union[str, Path]) -> str:
        return str(Path(filename).relative_to(self._base_folder))

    def keys(self) -> KeysView[str]:
        """Return all image keys"""
        return self._attrs.keys()

    @property
    def attrs(self) -> Dict[str, Dict[str, Any]]:
        """Return dict {key: attrs_dict}"""
        return self._attrs

    def items(self) -> ItemsView[str, xr.DataArray]:
        """Return list of (key, attributes) pairs"""
        return dict((key, self[key]) for key in self.keys()).items()

    def __getitem__(self, key: str) -> xr.DataArray:
        """Return image as DataArray from key"""
        da = self._get_image(self._base_folder / key)
        da.attrs.update(self._extract_attrs(key, relative=True))

        return da

    def iloc(self, i: int) -> xr.DataArray:
        """Return i-th image as DataArrray

        Raises IndexError if i >= len(self)
        """
        return self[self._filename2key(self._files[i])]

    def random(self) -> xr.DataArray:
        """Return random image as DataArray"""
        return self.iloc(np.random.randint(len(self)))

    def __iter__(self) -> Generator[str, None, None]:
        return (key for key in self.keys())

    def __contains__(self, key: Any) -> bool:
        return key in self._attrs

    def groupby(
        self, attr_name: str, sortby: Optional[Union[str, List[str]]] = None, ascending: bool = True
    ) -> "GroupedDataset":
        """Split dataset by a value of an attribute.

        Parameters
        ----------
        attr_name: str
            Name of the attribute used for splitting.
        sortby: str, list of str, optional
            Sort each group by the values of these attributes.
        ascending: bool
            Sort in ascending order?
        """
        sortby = sortby or []
        if isinstance(sortby, str):
            sortby = [sortby]

        if sortby:
            sort_fun = sorted
        else:

            def sort_fun(x: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: U100
                return x

        # build dict {attribute_value: sorted_list_of_keys_in_this_group}
        groups = defaultdict(lambda: [])
        for key, key_attrs in sort_fun(
            self.attrs.items(),
            key=lambda x: tuple(x[1][sort_col] for sort_col in sortby),  # type: ignore
            reverse=not ascending,
        ):
            groups[key_attrs.get(attr_name)].append(key)

        # return object representing the dataset split
        return GroupedDataset(
            self, key_groups=groups.values(), group_attrs=tuple({attr_name: k} for k in groups.keys())
        )


class GroupedDataset:
    def __init__(
        self,
        parent: Mapping[str, Any],
        key_groups: Iterable[List[str]],
        group_attrs: Sequence[Dict[str, Any]],
    ):
        """Dataset representing items from a parent dataset grouped by some condition.

        Usage:
            grouped_dataset[0] ... returns generator of all elements of i-th group
            for group in grouped_dataset:
                # group ~ grouped_dataset[i], i.e. it is a generator of all elemenents in the i-th group

        Parameters
        ----------
        parent: The parent dataset.
        key_groups: Iterable of lists of parent keys. Each list represents one group and contains
            all keys that belong to that group.
        group_attrs: Enumeration of attributes that defined the groups.
            E.g. [{product: 'IR108', date: '20201018}, {product: 'HRV', date: '20201018'},
                  {product: 'IR108', data: '20201019}]
        """
        self._parent = parent
        self._key_groups = list(key_groups)
        self._group_attrs = list(group_attrs)

        if len(self._key_groups) != len(self._group_attrs):
            raise ValueError(
                f"len(key_groups) != len(group_attrs): "
                f"{len(self._key_groups)} != {len(self._group_attrs)}"
            )

    def __len__(self) -> int:
        return len(self._key_groups)

    def __getitem__(self, i: int) -> Generator[xr.DataArray, Any, Any]:
        """Get all elements of the i-th group."""
        return (self._parent[key] for key in self._key_groups[i])

    @property
    def attrs(self) -> List[Dict[str, Any]]:
        return self._group_attrs

    def __iter__(self) -> Generator[Generator[xr.DataArray, None, None], None, None]:
        return (self[i] for i in range(len(self)))

    def filter(
        self,
        requested_groups: Optional[List[Dict[str, Any]]] = None,
        forbidden_groups: Optional[List[Dict[str, Any]]] = None,
        requested_attrs: Optional[Dict[str, Union[Any, Container[Any]]]] = None,
        forbidden_attrs: Optional[Dict[str, Union[Any, Container[Any]]]] = None,
    ) -> GroupedDataset:
        """Filter groups by the value of their attributes.

        Parameters
        ----------
        requested_groups: Return only groups with specified attributes and their values. Tests exact match of
            attribute dictionaries. Cannot be used simultaneously with forbidden_groups.
        forbidden_groups: Remove all groups with specified attributes and their values. Tests exact match of
            attribute dictionaries. Cannot be used simultaneously with requested_groups.
        requested_attrs: Return only groups whose attributes have values specified here. Each specified
            attribute must have one of the requested values.
        forbidden_attrs: Remove groups whose attributes have values specified here. Group is removed if any
            of the specified attributes has a forbidden value.

        Returns
        -------
        GroupedDataset
        """
        if all(
            param is None for param in [requested_groups, forbidden_groups, requested_attrs, forbidden_attrs]
        ):
            # no filter set
            return self

        if requested_groups is not None and forbidden_groups is not None:
            raise ValueError("Cannot use `requested_groups` and `forbidden_groups` at the same time.")

        requested_groups = requested_groups or []
        forbidden_groups = forbidden_groups or []
        requested_attrs = requested_attrs or {}
        forbidden_attrs = forbidden_attrs or {}

        requested_attrs = {k: tolist(v) for k, v in requested_attrs.items()}
        forbidden_attrs = {k: tolist(v) for k, v in forbidden_attrs.items()}

        filtered_group_attrs = []
        filtered_key_groups = []
        for group_attr, group_of_keys in zip(self._group_attrs, self._key_groups):
            if any(forbidden_group == group_attr for forbidden_group in forbidden_groups):
                # this group is forbidden
                continue

            if requested_groups and not any(
                allowed_group == group_attr for allowed_group in requested_groups
            ):
                # this group is not requested
                continue

            if any(
                attr in group_attr and group_attr[attr] in values for attr, values in forbidden_attrs.items()
            ):
                # this group contains forbidden value of an attribute
                continue

            if requested_attrs and not all(
                attr in group_attr and group_attr[attr] in values for attr, values in requested_attrs.items()
            ):
                # this group does not contain a requested value of an attribute
                continue

            filtered_group_attrs.append(group_attr)
            filtered_key_groups.append(group_of_keys)

        return GroupedDataset(self._parent, key_groups=filtered_key_groups, group_attrs=filtered_group_attrs)
