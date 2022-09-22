from typing import Optional
import os
from pathlib import Path

import importlib_resources as resources
import numpy as np
import pytest
import xarray as xr

from satdl.datasets import SatpyFolderDataset, SlotDefinition


@pytest.fixture
def slot_definition() -> SlotDefinition:
    with resources.as_file(resources.files('satdl') / 'definitions' / 'slot_MSG_CE.yaml') as path:
        yield SlotDefinition.from_yaml_file(path)


@pytest.fixture
def hrit_path() -> Path:
    path = os.environ.get('SATDL_TEST_HRIT_PATH')
    if not path:
        pytest.skip('$SATDL_TEST_HRIT_PATH not set')

    return Path(path)


@pytest.mark.parametrize('area', [(None, None), ('eurotv4n', (1152, 2048))])
def test_satpy_folder(hrit_path: Path, slot_definition: SlotDefinition, area: Optional[str]) -> None:
    ds = SatpyFolderDataset(hrit_path, slot_definition, area=area[0])

    # some data were actually loaded
    assert len(ds) > 0

    # there's more than one product available
    grid = ds.grid(['datetime', 'product'])
    assert len(grid.coords['product']) > 1
    assert len(grid.coords['datetime']) > 1

    # getitem returns array with some data
    for data_key in ds:  # TODO: in ds.values()
        if data_key[-1][1] in [
            'natural_color_with_night_ir_hires',
            'natural_enh_with_night_ir_hires',
            'night_ir_with_background_hires'
        ]:
            # this composite is too memory intensive -> skip
            continue
        data = ds[data_key]
        assert data is not None, f"Failed converting {data_key} to data."
        assert isinstance(data, xr.DataArray)
        assert np.isfinite(data).any()
        assert data.ndim == 3
        assert set(data.dims) == {'bands', 'x', 'y'}
        if area[1]:
            assert data.shape ==(3, *area[1])
        else:
            assert data.shape[0] == 3

        del data
