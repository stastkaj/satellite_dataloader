from typing import Generator, Optional, Tuple
from logging import getLogger
import os
from pathlib import Path
import random

import importlib_resources as resources
import numpy as np
import pytest
import xarray as xr

from satdl.datasets import SatpyFolderDataset, SlotDefinition


logger = getLogger()


@pytest.fixture
def slot_definition() -> Generator[SlotDefinition, None, None]:
    with resources.as_file(resources.files("satdl") / "definitions" / "slot_MSG_CE.yaml") as path:
        yield SlotDefinition.from_yaml_file(path)


@pytest.fixture
def hrit_path() -> Path:
    path = os.environ.get("SATDL_TEST_HRIT_PATH")
    if not path:
        pytest.skip("$SATDL_TEST_HRIT_PATH not set")

    return Path(path)


@pytest.mark.parametrize("area_tuple", [(None, None), ("eurotv4n", (1152, 2048))], ids=['no_projection', 'projection'])
def test_satpy_folder(
    hrit_path: Path, slot_definition: SlotDefinition, area_tuple: Tuple[Optional[str], Optional[Tuple[int, int]]]
) -> None:
    area, area_shape = area_tuple
    ds = SatpyFolderDataset(hrit_path, slot_definition, area=area, max_cache=0) # turn off caching to save memory

    # some data were actually loaded
    assert len(ds) > 0

    # there's more than one product available
    grid = ds.grid(["datetime", "product"])
    assert len(grid.coords["product"]) > 1
    assert len(grid.coords["datetime"]) > 1

    # getitem returns array with some data - test on 2 random products
    memory_intensive_data_keys = [
        "natural_color_with_night_ir_hires",
        "natural_enh_with_night_ir_hires",
        "night_ir_with_background_hires",
    ]  # this composite is too memory intensive -> skip
    data_keys = [data_key for data_key in list(ds) if data_key.split("|")[0] not in memory_intensive_data_keys]
    data_keys = random.choices(data_keys, k=2)

    for data_key in data_keys:
        data = ds[data_key]
        assert data is not None, f"Failed converting {data_key} to data."
        assert isinstance(data, xr.DataArray)
        assert np.isfinite(data).any()
        assert data.ndim == 3
        assert set(data.dims) == {"bands", "x", "y"}
        if area_shape:
            assert data.shape == (3, *area_shape)
        else:
            assert data.shape[0] == 3
        # has lon, lat coordinates
        for coord in ["lon", "lat"]:
            assert coord in data.coords
            assert data.coords[coord].dims == ("y", "x")
        logger.info(f'Tested {data_key.split("|")[0]} product.')


@pytest.mark.parametrize("area", ["eurotv4n"], ids=['projection'])
def test_satpy_folder_caching(
    hrit_path: Path, slot_definition: SlotDefinition, area: Optional[str]
) -> None:
    import time
    ds = SatpyFolderDataset(hrit_path, slot_definition, area=area, max_cache=50) # turn on caching

    # getitem returns array with some data - test on 2 random products
    memory_intensive_data_keys = [
        "natural_color_with_night_ir_hires",
        "natural_enh_with_night_ir_hires",
        "night_ir_with_background_hires",
    ]  # this composite is too memory intensive -> skip
    data_keys = [data_key for data_key in list(ds) if data_key.split("|")[0] not in memory_intensive_data_keys]
    data_key = random.choice(data_keys)

    t1 = time.perf_counter_ns()
    im = ds[data_key]
    assert isinstance(im, xr.DataArray)
    dt1 = time.perf_counter_ns() - t1

    # second run must be faster than the first one thanks to memory caching
    t2 = time.perf_counter_ns()
    im = ds[data_key]
    assert isinstance(im, xr.DataArray)
    dt2 = time.perf_counter_ns() - t2

    assert dt2 < dt1
