from __future__ import annotations

from typing import (
    Any,
    Callable,
    Container,
    DefaultDict,
    Dict,
    Generator,
    Generic,
    ItemsView,
    Iterable,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)
from collections import abc, defaultdict
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from trollsift import Parser
import xarray as xr

from satdl.utils import image2xr, tolist


_logger = logging.getLogger(__name__)


DataType = TypeVar("DataType")


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
        self._ind2key = {ind: key for ind, key in enumerate(self._attrs)}

        self._get_image = lru_cache(max_cache)(_get_get_image(self._georef))

    def __len__(self) -> int:
        return len(self._files)

    def _extract_attrs(self, filename: Union[str, Path], relative: bool = False) -> Dict[str, Any]:
        key = str(filename) if relative else self._filename2key(filename)
        return self._file_mask.parse(key)

    def _filename2key(self, filename: Union[str, Path]) -> str:
        return str(Path(filename).relative_to(self._base_folder))

    def keys(self) -> KeysView[str]:
        """Return all image keys."""
        return self._attrs.keys()

    @property
    def attrs(self) -> Dict[str, Dict[str, Any]]:
        """Return dict {key: attrs_dict}."""
        return self._attrs

    @overload
    def get_attrs(self, key: Union[int, str]) -> Dict[str, Any]:  # noqa: U100
        ...

    @overload
    def get_attrs(self, key: List[Union[int, str]]) -> List[Dict[str, Any]]:  # noqa: U100
        ...

    def get_attrs(
        self, key: Union[int, str, List[Union[int, str]]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Return attributes of i-th data element.

        If key is an integer, it is interpreted as a positional index.

        If key is a sequence, a list of attrs is returned.
        """
        if not isinstance(key, str):
            if isinstance(key, abc.Sequence):
                # return sequence
                return [self.get_attrs(k) for k in key]

            # convert index to key
            key = self._ind2key[key]

        return self._attrs[key]

    def items(self) -> ItemsView[str, xr.DataArray]:
        """Return list of (key, attributes) pairs."""
        return dict((key, self[key]) for key in self.keys()).items()

    @overload
    def __getitem__(self, key: Union[int, str]) -> xr.DataArray:  # noqa: U100
        ...

    @overload
    def __getitem__(self, key: List[Union[int, str]]) -> List[xr.DataArray]:  # noqa: U100
        ...

    def __getitem__(
        self, key: Union[int, str, List[Union[int, str]]]
    ) -> Union[xr.DataArray, List[xr.DataArray]]:
        """Return image as DataArray from key.

        If key is an integer, it is interpreted as a positional index.

        If key is a sequence, a list of DataArrays is returned.
        """
        if not isinstance(key, str):
            if isinstance(key, abc.Sequence):
                # return sequence
                return [self[k] for k in key]

            # convert index to key
            key = self._ind2key[key]

        da = self._get_image(self._base_folder / key)
        da.attrs.update(self._extract_attrs(key, relative=True))

        return da

    def iloc(self, i: int) -> xr.DataArray:
        """Return i-th image as DataArrray.

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
        self,
        group_attr: str,
        align: Optional[Union[str, Sequence[str]]] = None,
        group_ascending: Optional[bool] = None,
        align_ascending: Optional[bool] = None,
    ) -> "GroupedDataset":
        """Split dataset by a value of an attribute.

        Parameters
        ----------
        group_attr: str
            Name of the attribute used for splitting.
        sortby: str, list of str, optional
            Sort each group by the values of these attributes.
        align: Attribute or collection of attributes that should be aligned into a grid. Cannot
            align by group attrs. If used, all groups are guaranteed to have the same length.
            If some combination of align attributes is missing in some groups, its data will
            be None. For instance, `groupby('product', sortby: 'time', align='time')` will guarantee
            that each group will have the same length equal to the number of unique times in the data
            and if data for a particular time are not available in some group, it will return None.
        group_ascending: bool, optional
            Sort groups? Do not sort if None.
        align_ascending: bool, optional
            Sort by the align attributes in ascending order? Do not sort if None.
        """
        align = tolist(align)

        if group_attr in align:
            raise ValueError("`align` contains groupby `group_attr`.")

        important_attrs = {group_attr} | set(align)

        # hack to avoid casting of attr values by pandas to a different type, e.g. datetime to np.datetime
        # raises exception if values of important_attrs are not hashable
        # replaces attr values by an integer index
        # TODO: avoid this hack?
        # TODO: generalize for unhashable values
        attr2index: DefaultDict[str, Dict[Any, int]] = defaultdict(dict)
        index2attr: DefaultDict[str, Dict[int, Any]] = defaultdict(dict)
        attrs = []
        for data_key, data_attrs in self.attrs.items():
            attrs_dict = {}
            for attr, attr_value in data_attrs.items():
                if attr not in important_attrs:
                    continue

                if attr_value not in attr2index[attr]:
                    ind = len(attr2index[attr])
                    attr2index[attr][attr_value] = ind
                    index2attr[attr][ind] = attr_value
                else:
                    ind = attr2index[attr][attr_value]

                attrs_dict[attr] = ind

            attrs.append(attrs_dict)

        data_key_column = "__data_key__"
        while data_key_column in important_attrs:  # avoid conflict with attribute names
            data_key += "_"

        df = pd.DataFrame.from_dict(attrs)
        df[data_key_column] = self.keys()

        df_pivot = df.pivot(index=group_attr, columns=align, values=data_key_column)

        # sort align attributes by value
        if align_ascending is not None:

            def col2sortkey(col: Any) -> Sequence[Any]:
                col = tolist(col)
                return tuple(index2attr[attr][col[i]] for i, attr in enumerate(align))  # type: ignore

            sorted_cols = sorted(
                df_pivot.columns, key=col2sortkey, reverse=not align_ascending  # type: ignore
            )
            df_pivot = df_pivot[sorted_cols]

        # sort group attributes by value
        if group_ascending is not None:
            sorted_indices = [
                ind
                for _, ind in sorted(
                    ((index2attr[group_attr][i], i) for i in df_pivot.index), reverse=not group_ascending
                )
            ]
            df_pivot = df_pivot.iloc[sorted_indices]

        # return object representing the dataset split
        group_attrs = [{group_attr: index2attr[group_attr][ind]} for ind in df_pivot.index.values]
        group_keys = [group_keys.values for _, group_keys in df_pivot.iterrows()]
        return GroupedDataset(self, group_keys=group_keys, group_attrs=group_attrs)

    def index_grid(self, dims: Sequence[str], ascending: Optional[bool] = None) -> xr.DataArray:
        """Interpret given attributes as coordinates and return grid of data indices in these coordinates.

        Missing data are marked by a negative number.

        Parameters
        ----------
        dims: List of attributes that will become grid dimensions.

        Returns
        -------
        xarray DataArray with indices of images
        """
        # find unique values of coordiantes
        coords = defaultdict(set)
        for data_attrs in self.attrs.values():
            for attr in dims:
                coords[attr].add(data_attrs.get(attr, None))

        # build the grid
        grid = xr.DataArray(dims=dims, coords={k: list(v) for k, v in coords.items()}).astype(int)
        grid[:] = -1  # -1 means no data
        for ind, data_attrs in enumerate(self._attrs.values()):
            c = {dim: data_attrs.get(dim, None) for dim in dims}
            if grid.sel(c) >= 0:
                raise ValueError(f"Found duplicate data items for grid coords: {c}.")

            grid.loc[c] = ind

        # sort the coordinates
        if ascending is not None:
            grid = grid.sortby(dims, ascending=ascending)

        return grid


class GroupedDataset(Generic[DataType]):
    def __init__(
        self,
        parent: Mapping[str, DataType],
        group_keys: Iterable[List[str]],
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
        group_keys: Iterable of lists of parent keys. Each list represents one group and contains
            all keys that belong to that group.
        group_attrs: Enumeration of attributes that defined the groups.
            E.g. [{product: 'IR108', date: '20201018}, {product: 'HRV', date: '20201018'},
                  {product: 'IR108', data: '20201019}]
        """
        self._parent = parent
        self._group_keys = list(group_keys)
        self._group_attrs = list(group_attrs)

        if len(self._group_keys) != len(self._group_attrs):
            raise ValueError(
                f"len(group_keys) != len(group_attrs): "
                f"{len(self._group_keys)} != {len(self._group_attrs)}"
            )

    def __len__(self) -> int:
        return len(self._group_keys)

    def __getitem__(self, i: int) -> Generator[Optional[DataType], Any, Any]:
        """Get all elements of the i-th group."""
        return (self._parent[key] for key in self._group_keys[i])

    @property
    def attrs(self) -> List[Dict[str, Any]]:
        """Return list of attributes that were used to create the groups."""
        return self._group_attrs

    def __iter__(self) -> Generator[Generator[Optional[DataType], None, None], None, None]:
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
        filtered_group_keys = []
        for group_attr, group_of_keys in zip(self._group_attrs, self._group_keys):
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
            filtered_group_keys.append(group_of_keys)

        return GroupedDataset(self._parent, group_keys=filtered_group_keys, group_attrs=filtered_group_attrs)
