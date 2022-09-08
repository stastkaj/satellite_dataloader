import numpy as np
import pytest
import xarray as xr

from satdl.datasets import GriddedDataset


@pytest.fixture
def gridded_dataset() -> GriddedDataset:
    data = [0, 1, 2, 3, 4]
    index_grid = xr.DataArray(
        np.array([[0, 2, 1], [-1, 4, 3]]).T, dims=["a", "b"], coords={"a": [0, 1, 2], "b": ["0", "1"]}
    )

    def is_negative(x: int) -> bool:
        return x < 0

    return GriddedDataset(data, index_grid, invalid_key=is_negative)


def test_gridded_dataset_invalid_key(gridded_dataset: GriddedDataset) -> None:
    # invalid_key can be a collable
    assert callable(gridded_dataset.invalid_key)
    assert gridded_dataset[0, "1"] is None
    # invalid_key can be a value
    grid_ds = GriddedDataset(gridded_dataset.dataset, gridded_dataset.index_grid, invalid_key=-1)
    assert grid_ds[0, "1"] is None


def test_gridded_dataset(gridded_dataset: GriddedDataset) -> None:
    # has correct dimensions
    assert gridded_dataset.ndim == 2
    assert gridded_dataset.dims == ("a", "b")
    assert gridded_dataset.iter_dim == "a"

    # sel works
    sel_dataset = gridded_dataset.sel(a=1)
    assert (sel_dataset.index_grid == gridded_dataset.index_grid.sel(a=1)).all()
    with pytest.raises(KeyError):
        gridded_dataset.index_grid.sel(a=5)

    # isel works
    sel_dataset = gridded_dataset.isel(b=0)
    assert (sel_dataset.index_grid == gridded_dataset.index_grid.isel(b=0)).all()
    with pytest.raises(IndexError):
        gridded_dataset.index_grid.isel(b=5)

    # single iter returns another gridded dataset
    iterlist = [v for v in gridded_dataset]
    assert all(v.dims == ("b",) for v in iterlist)
    assert len(iterlist) == len(gridded_dataset.index_grid.coords["a"])
    assert iterlist[0].index_grid.values.tolist() == [0, -1]
    assert iterlist[2].index_grid.values.tolist() == [1, 3]

    # double iter returns data
    doubleiterlist = [v for v in iterlist[0]]
    assert doubleiterlist == [0, None]

    # single get item returns gridded dataset
    grid_b = gridded_dataset[1]
    assert (grid_b.index_grid == gridded_dataset.index_grid.sel(a=1)).all()

    # double get item returns values
    assert grid_b["0"] == 2
    assert grid_b["1"] == 4
    assert gridded_dataset[0, "0"] == 0
    assert gridded_dataset[0, "1"] is None

    # too many coords raises ValueError
    with pytest.raises(ValueError):
        gridded_dataset[1, "0", 1]

    # non-existing coordinates raise KeyError
    with pytest.raises(KeyError):
        gridded_dataset[10]

    # non-unique coordinates raise ValueError
    with pytest.raises(ValueError):
        gridded_dataset[slice(None, None)]
