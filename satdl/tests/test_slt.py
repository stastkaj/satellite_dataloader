from satdl.datasets import ImageFolderDataset
from satdl.utils import image2xr


_data_path = "/home/seidl/git/satellite-labelling-tool/slt/examples/images"
_image_file_mask = "{projection}-{resolution}.{product}.{datetime:%Y%m%d.%H%M}.0.jpg"
_georef_path = _data_path + "/201911271130_MSG4_msgce_1160x800_geotiff_hrv.tif"
_georef = image2xr(_georef_path)


def test_slt() -> None:
    image_dataset = ImageFolderDataset(_data_path, file_mask=_image_file_mask, georef=_georef, max_cache=100)
    # use xarray to create rectangular grid of data indexes with coordinates product and datetime
    image_dataloader = image_dataset.grid(["product", "datetime"], ascending=True)

    dl = image_dataloader.isel(datetime=0)
    dl["hrv"]
