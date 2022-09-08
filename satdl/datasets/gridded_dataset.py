from __future__ import annotations

from typing import Any, Callable, Hashable, Iterator, Mapping, Optional, Sequence, Tuple, Union
from functools import wraps
import logging

import xarray as xr

from satdl.utils import tolist


_logger = logging.getLogger(__name__)


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
