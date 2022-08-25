from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from satdl.datasets import StaticImageFolderDataset


FIXTURE_DIR = Path(__file__).parent / "test_data"


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd(datafiles) -> None:  # type: ignore
    # can build the object
    sifd = StaticImageFolderDataset(
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
    assert (sifd[3] == sifd[list(sifd.keys())[3]]).all()  # type: ignore
    assert sifd.get_attrs(3) == sifd.get_attrs(list(sifd.keys())[3])

    # indexing by a sequence works
    sifd3, sifd2 = sifd[[3, 2]]
    assert (sifd3 == sifd[list(sifd.keys())[3]]).all()
    assert (sifd2 == sifd[list(sifd.keys())[2]]).all()

    sifd3a, sifd2a = sifd.get_attrs([3, 2])
    assert sifd3a == sifd.get_attrs(list(sifd.keys())[3])
    assert sifd2a == sifd.get_attrs(list(sifd.keys())[2])

    sifd3, sifd2 = sifd[[3, list(sifd.keys())[2]]]
    assert (sifd3 == sifd[list(sifd.keys())[3]]).all()
    assert (sifd2 == sifd[list(sifd.keys())[2]]).all()

    sifd3a, sifd2a = sifd.get_attrs([3, list(sifd.keys())[2]])
    assert sifd3a == sifd.get_attrs(list(sifd.keys())[3])
    assert sifd2a == sifd.get_attrs(list(sifd.keys())[2])


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_iter(datafiles) -> None:  # type: ignore
    sifd = StaticImageFolderDataset(
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
    sifd = StaticImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        max_cache=50,
    )

    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)

    for key in sifd.keys():
        im = sifd[key]

        assert isinstance(im, xr.DataArray)


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_groupby(datafiles) -> None:  # type: ignore
    sifd = StaticImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        max_cache=None,
    )

    # groupby works for string align and ascending=True
    groups = sifd.groupby("datetime", align="product", align_ascending=True, group_ascending=False)

    assert len(groups) == 3
    for group in groups:
        assert len(list(group)) == 4
    # group keys should be sorted in ascending order
    for group in groups:
        assert [da.attrs["product"] for da in group] == sorted(  # type: ignore
            set([attr["product"] for attr in sifd.attrs.values()])
        )
    # groups should be sorted in descending order
    assert [attr["datetime"] for attr in groups.attrs] == sorted(
        set([attr["datetime"] for attr in sifd.attrs.values()]), reverse=True
    )

    # groupby works for list align and ascending=False
    groups = sifd.groupby(
        "product", align=["datetime", "projection"], align_ascending=False, group_ascending=True
    )

    assert len(groups) == 4
    for group in groups:
        assert len(list(group)) == 3
    # group keys should be sorted in ascending order
    for group in groups:
        assert [da.attrs["datetime"] for da in group] == sorted(  # type: ignore
            set([attr["datetime"] for attr in sifd.attrs.values()]), reverse=True
        )
    # groups should be sorted in asscending order
    assert [attr["product"] for attr in groups.attrs] == sorted(
        set([attr["product"] for attr in sifd.attrs.values()]), reverse=False
    )


@pytest.mark.datafiles(FIXTURE_DIR / "images")
@pytest.mark.datafiles(FIXTURE_DIR / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif")
def test_sifd_grid(datafiles) -> None:  # type: ignore
    sifd = StaticImageFolderDataset(
        datafiles,
        "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg",
        georef=Path(datafiles) / "201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif",
        max_cache=None,
    )

    grid = sifd.index_grid(["product", "datetime"], ascending=False)

    # grid contains unique indicies 0..12
    assert grid.dtype == int
    assert len(np.unique(grid)) == 12
    assert grid.min() == 0
    assert grid.max() == 11

    # assert the assignment is correct
    for ind in grid.sel(product="vis-ir").values.ravel():
        assert sifd.get_attrs(ind)["product"] == "vis-ir"  # type: ignore

    # axes are sorted
    assert grid.product.values.tolist() == sorted(grid.product.values.tolist(), reverse=True)
