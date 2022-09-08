from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Hashable,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from collections import abc, defaultdict
from functools import lru_cache, wraps
import logging
from pathlib import Path

import numpy as np
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
        """Dataset of georeferenced images in a local folder.

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
        for ind, data_attrs in enumerate(self._attrs.values()):
            c = {dim: data_attrs.get(dim, None) for dim in dims}
            if grid.sel(c) >= 0:
                raise ValueError(f"Found duplicate data items for grid coords: {c}.")

            grid.loc[c] = ind

        # sort the coordinates
        if ascending is not None:
            grid = grid.sortby(dims, ascending=ascending)

        return grid

    def grid(self, dims: Sequence[str], ascending: Optional[bool] = None) -> "GriddedDataset":
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
            self, self.index_grid(dims=dims, ascending=ascending), invalid_key=lambda x: x < 0
        )


class GriddedDataset:
    """Dataset on a grid given by some orthogonal axes.

    Args:
        dataset: Dataset to be wrapped.
        index_grid: Grid of dataset keys.
        dim_order: New order of index_grid dimensions, optional. Does not need
            to contain all dimensions, just the ordering of those that should be
            moved to the beginning. The rest is not changed. When iterating over
            the gridded dataset, successive iterators are created in the order
            given by index_grid dimensions.
        invalid_key: Value or callable that decides which key in the index_grid
            is treated as a marker for missing data.
    """

    def __init__(
        self,
        dataset: Union[Sequence, Mapping],
        index_grid: xr.DataArray,
        dim_order: Optional[Union[str, Sequence[str]]] = None,
        invalid_key: Union[Callable[[Any], bool], Any] = lambda x: False,
    ):
        dim_order = tolist(dim_order)
        self.dataset = dataset
        self.index_grid = index_grid.transpose(*dim_order, ...)
        self.invalid_key = invalid_key

    @property
    def iter_dim(self) -> Hashable:
        """Name of the first dimension over which the grid will be iterated."""
        return self.index_grid.dims[0]

    @property
    def dims(self) -> Tuple[Hashable, ...]:
        """List of dimensions of the grid."""
        return self.index_grid.dims

    @property
    def coords(self) -> xr.core.coordinates.DataArrayCoordinates:
        return self.index_grid.coords

    def __len__(self) -> int:
        return len(self.index_grid.coords[self.index_grid.dims[0]])

    # TODO: replace Any with type of data in the dataset
    def __iter__(self) -> Iterator[Union["GriddedDataset", Any]]:
        return (self[coord] for coord in self.index_grid.coords[self.iter_dim])

    # TODO: replace Any with type of data in the dataset
    def iterby(self, dim: Union[str, Sequence[str]]) -> Iterator[Union["GriddedDataset", Any]]:
        """Iterate by given dimension.

        Args:
            dim: Dimension along which to iterate. If list is given, sets also
                ordering of dimensions for sub-iterators.
        """
        return iter(
            GriddedDataset(self.dataset, self.index_grid, dim_order=dim, invalid_key=self.invalid_key)
        )

    # TODO: replace Any with type of data in the dataset
    def __getitem__(self, key: Any) -> Union["GriddedDataset", Any]:
        """Cut the grid at selected coordinate value(s). Return data if all dimensions have been eliminated.

        Args:
            *args: Represent coordinate values for index_grid dimensions in their respecective order.
                Can be anything accepted by xr.DataArray.sel, but must correspond to exactly one coordinate
                value for each dimension.

        Returns:
            GriddedData with first len(args) dimensions eliminated or data item from the dataset.

        Raises:
            ValueError: More arguments than dimensions or non-unique coordinate range.
            KeyError: Wrong coordinates.
        """
        args = tolist(key)

        if len(args) == 0:
            return self

        index_subgrid = self.index_grid.sel(indexers={self.iter_dim: args[0]})
        if index_subgrid.coords[self.iter_dim].ndim > 0:
            raise ValueError(f"Multiple coordinate values left for dimension {self.iter_dim}")
        index_subgrid = index_subgrid.drop_vars(self.iter_dim)

        if self.ndim > 1:
            # more than one dimension => return GriddedDataset
            return GriddedDataset(self.dataset, index_subgrid, invalid_key=self.invalid_key)[args[1:]]

        # last dimension => return data
        if len(args) > 1:
            raise ValueError("Too many arguments. Got more arguments than dimensions.")

        key = index_subgrid.values.item()
        if callable(self.invalid_key):
            if self.invalid_key(key):
                return None
        elif self.invalid_key == key:
            return None

        return self.dataset[key]

    @property
    def ndim(self) -> int:
        """Number of dimensions of the grid."""
        return self.index_grid.ndim

    @wraps(xr.DataArray.sel)
    def sel(self, *args: Any, **kwargs: Any) -> "GriddedDataset":
        """Wrapper around xarray sel with the same arguments."""
        return GriddedDataset(
            self.dataset, index_grid=self.index_grid.sel(*args, **kwargs), invalid_key=self.invalid_key
        )

    @wraps(xr.DataArray.isel)
    def isel(self, *args: Any, **kwargs: Any) -> "GriddedDataset":
        """Wrapper around xarray isel with the same arguments.

        Raises:
            IndexError
        """
        return GriddedDataset(
            self.dataset, index_grid=self.index_grid.isel(*args, **kwargs), invalid_key=self.invalid_key
        )
