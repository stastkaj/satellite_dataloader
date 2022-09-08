from typing import Any, Dict, Generic, Mapping, Optional, Sequence, TypeVar
from abc import ABC, abstractmethod
from collections import defaultdict

import xarray as xr

from satdl.datasets import GriddedDataset


DataType = TypeVar("DataType")


class AttributeDatasetBase(Generic[DataType], Mapping[str, DataType], ABC):
    @property
    @abstractmethod
    def attrs(self) -> Dict[str, Dict[str, Any]]:
        """Return dict {key: attrs_dict}."""

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
            self, self.index_grid(dims=dims, ascending=ascending), invalid_key=lambda x: x < 0
        )
