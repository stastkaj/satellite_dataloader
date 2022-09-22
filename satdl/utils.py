from typing import Any, Callable, List, Optional, Union
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
import xarray as xr


def image2xr(
    path: Union[str, Path],
    georef: Optional[Union[str, Path, xr.DataArray]] = None,
    require_georef: bool = True,
) -> xr.DataArray:
    """Load (possibly georeferenced) image

    Parameters
    ----------
    path : str or Path
        path to image to be loaded
    georef: str or Path or xr.DataArray
        georeference using this image or DataArray (must contain coordinates lon
        and lat and have the same size)
    require_georef : bool
        raise if the output misses lon/lat coordinates? (default: True)
    """
    da = xr.open_rasterio(path)

    if georef is not None:
        if not isinstance(georef, xr.DataArray):
            georef = image2xr(georef)

        da.coords["x"] = georef.x.data
        da.coords["y"] = georef.y.data
        da.coords["lat"] = (("y", "x"), georef.lat.data)
        da.coords["lon"] = (("y", "x"), georef.lon.data)

        for attr in ["transform", "res", "crs"]:
            da.attrs[attr] = georef.attrs.get(attr)

    elif "crs" in da.attrs:
        da_crs = CRS.from_proj4(da.attrs["crs"])
        wgs84 = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(da_crs, wgs84)

        xx, yy = np.meshgrid(da.x, da.y)
        lat, lon = transformer.transform(xx, yy)

        da.coords["lat"] = (("y", "x"), lat)
        da.coords["lon"] = (("y", "x"), lon)

    if require_georef and ("lat" not in da.coords or "lon" not in da.coords or "crs" not in da.attrs):
        raise ValueError(f"cannot georeference image {path}")

    return da


def tolist(x: Any, none_as_empty_list: bool = True, converter: Optional[Callable[[Any], Any]] = None) -> List[Any]:
    """Convert to list.

    Optionally convert all elements using the converter function.
    """
    if none_as_empty_list and x is None:
        return []

    if isinstance(x, (list, tuple, set)):
        if not converter:
            return list(x)
        else:
            return [converter(element) for element in x]
    else:
        #
        if not converter:
            return [x]
        else:
            return [converter(x)]
