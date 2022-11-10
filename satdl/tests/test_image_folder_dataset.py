from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from satdl.datasets import ImageFolderDataset


FIXTURE_DIR = Path(__file__).parent / "test_data"


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd(datafiles) -> None:  # type: ignore
    # can build the object
    sifd = ImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
    )

    # sifd contains all files it should
    assert len(sifd) == 12
    for f in Path(datafiles).glob("*.jpg"):
        assert Path(f).name in sifd.keys()

    # items has correct attributes
    for attr in sifd.attrs.values():
        assert attr["projection"] == "msgce"

    # can load the dataa
    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)
        assert im.shape == (3, 800, 1160)
        assert im.lat.min() > 42.27232
        assert im.lat.max() < 56.64341
        assert im.lon.min() > -1.687554
        assert im.lon.max() < 30.715534

    # indexing by integer works
    assert (sifd.iloc[3] == sifd[list(sifd.keys())[3]]).all()  # type: ignore
    for data, index in zip(sifd.iloc[3:5], [3, 4]):
        assert (data == sifd[list(sifd.keys())[index]]).all()  # type: ignore


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_iter(datafiles) -> None:  # type: ignore
    sifd = ImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
    )
    i = 0
    for key in sifd:
        assert (Path(datafiles) / key).exists()
        i += 1

    assert i == 12


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_cache(datafiles) -> None:  # type: ignore
    import time

    sifd = ImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        cache=lru_cache(50),
    )

    t1 = time.perf_counter_ns()
    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)
    dt1 = time.perf_counter_ns() - t1

    t2 = time.perf_counter_ns()
    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)
    dt2 = time.perf_counter_ns() - t2

    assert dt2 < dt1


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_index_grid(datafiles) -> None:  # type: ignore
    sifd = ImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        cache=None,
    )

    grid = sifd.index_grid(["product", "datetime"], ascending=False)

    # grid contains unique indicies 0..12
    assert grid.dtype == int
    assert len(np.unique(grid)) == 12
    assert grid.min() == 0
    assert grid.max() == 11

    # assert the assignment is correct
    for ind in grid.sel(product="vis-ir").values.ravel():
        assert sifd.attrs[list(sifd.keys())[ind]]["product"] == "vis-ir"  # type: ignore

    # axes are sorted
    assert grid.product.values.tolist() == sorted(grid.product.values.tolist(), reverse=True)


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_grid(datafiles) -> None:  # type: ignore
    sifd = ImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        cache=None,
    )

    grid = sifd.grid(["product", "datetime"], ascending=False)

    assert grid.dataset is sifd.iloc
    assert (grid.index_grid == sifd.index_grid(["product", "datetime"], ascending=False)).all()
