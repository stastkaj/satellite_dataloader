from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    ItemsView,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import xarray as xr

from satdl.datasets import GriddedDataset


ItemType = TypeVar("ItemType")
KeyType = TypeVar("KeyType")
DataType = TypeVar("DataType")


class AttributeDatasetBase(Generic[ItemType, KeyType, DataType], Mapping[KeyType, DataType], ABC):
    def __init__(self) -> None:
        self.iloc = ILoc(self)
        self._items = self._find_items()
        self._attrs = {self._item2key(f): self._extract_attrs(f) for f in self._items}
        self._ind2key = {ind: key for ind, key in enumerate(self._attrs)}
        self._key2ind = {key: ind for ind, key in enumerate(self._attrs)}

    @abstractmethod
    def _find_items(self) -> List[ItemType]:
        """Find all items in the dataset."""

    @abstractmethod
    def _extract_attrs(self, item: ItemType) -> Dict[str, Any]:  # noqa: U100
        """"""

    @abstractmethod
    def _item2key(self, item: ItemType) -> KeyType:  # noqa: U100
        """Convert item to its key."""

    def _key2item(self, key: KeyType) -> ItemType:  # noqa: U100
        """Convert key to corresponding item."""
        return self._items[self._key2ind[key]]

    @abstractmethod
    def _get_data(self, key: KeyType) -> DataType:  # noqa: U100
        """Convert key to data, return None if not possible."""

    def __len__(self) -> int:
        return len(self._items)

    def keys(self) -> KeysView[KeyType]:
        """Return all image keys."""
        return self._attrs.keys()

    def items(self) -> ItemsView[KeyType, DataType]:
        """Return list of (key, attributes) pairs."""
        return dict((key, self[key]) for key in self.keys()).items()

    @property
    def attrs(self) -> Dict[KeyType, Dict[str, Any]]:
        """Return dict {key: attrs_dict}."""
        return self._attrs

    def random(self) -> DataType:
        """Return random image as DataArray"""
        return self.iloc[np.random.randint(len(self))]

    def __iter__(self) -> Generator[KeyType, None, None]:
        return (key for key in self.keys())

    def __contains__(self, key: Any) -> bool:
        return key in self._attrs

    def __getitem__(self, key: KeyType) -> DataType:
        """Return data given key."""
        return self._get_data(key)

    def index_grid(self, dims: Sequence[str], ascending: Optional[bool] = None) -> xr.DataArray:
        """Interpret given attributes as coordinates and return grid of data indices in these coordinates.

        Missing data are marked by a negative number.

        Parameters
        ----------
        dims: List of attributes that will become grid dimensions.
        ascending: Sort axes in ascending (True), descending(False) order or not sort (None).

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
        for ind, data_attrs in enumerate(self.attrs.values()):
            c = {dim: data_attrs.get(dim, None) for dim in dims}
            if grid.sel(c) >= 0:
                raise ValueError(f"Found duplicate data items for grid coords: {c}.")

            grid.loc[c] = ind

        # sort the coordinates
        if ascending is not None:
            grid = grid.sortby(dims, ascending=ascending)

        return grid

    def grid(self, dims: Sequence[str], ascending: Optional[bool] = None) -> GriddedDataset:
        """Return gridded dataset for given dimensions.

        Parameters
        ----------
        dims: List of attributes that will become grid dimensions.
        ascending: Sort axes in ascending (True), descending(False) order or not sort (None).

        Returns
        -------
        GriddedDataset
        """
        return GriddedDataset(
            self.iloc, self.index_grid(dims=dims, ascending=ascending), invalid_key=lambda x: x < 0
        )


class ILoc(Sequence[DataType]):
    def __init__(self, obj: AttributeDatasetBase[ItemType, KeyType, DataType]) -> None:
        self.obj = obj

    def __len__(self) -> int:
        return len(self.obj)

    @overload
    def __getitem__(self, index: int) -> DataType:  # noqa: U100
        ...

    @overload
    def __getitem__(self, index: slice) -> List[DataType]:  # noqa: U100
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[DataType, List[DataType]]:
        """Return data of i-th element.

        Raises IndexError if i >= len(obj)
        """
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            return self.obj[self.obj._item2key(self.obj._items[index])]
        else:
            raise TypeError(f"'index' must be int or slice, got {type(index)}.")
