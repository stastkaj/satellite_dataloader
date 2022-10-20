from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from functools import lru_cache, partial
import logging
from pathlib import Path

from attrs import frozen
from pyresample import AreaDefinition
from satpy.writers import get_enhanced_image
import xarray as xr

from satdl.datasets._dataset_base import AttributeDatasetBase
from satdl.datasets._segment_gatherer import SatpySlotFiles, SegmentGatherer, SlotDefinition


_logger = logging.getLogger(__name__)


@frozen
class SatpyProductFiles:
    slot_files: SatpySlotFiles
    product: str


class SatpyFolderDataset(AttributeDatasetBase[SatpyProductFiles, str, xr.DataArray]):
    _forbidden_composites_wo_area = [
        "ir108_3d",
        "natural_color_with_night_ir",
        "natural_enh_with_night_ir",
        "night_ir_with_background",
    ]  # TODO: find why these do not work, move the list to some config file

    def __init__(
        self,
        base_path: Union[str, Path],
        slot_definition: SlotDefinition,
        area: Optional[Union[str, AreaDefinition]],
        max_cache: Optional[int] = None,
    ) -> None:
        """Dataset of satpy products in a local folder.

        Note: content of the folder is scanned only once, at the class creation

        Parameters
        ----------
        base_path:
            Root folder of the data.
        slot_definition:
            Definition of satpy files belonging to a single slot.
        area:
            Projection of the resulting image. Optional.
        max_cache: int, optional
            Maximum number of images that will be cached.
        """
        self._base_path = Path(base_path)
        self._slot_definition = slot_definition
        self._area = area

        super().__init__()

        if max_cache is None:
            max_cache = 0
        self._get_image = lru_cache(max_cache)(partial(self._data2image, area=area))

    def _data2image(
        self, item: SatpyProductFiles, area: Optional[Union[str, AreaDefinition]]
    ) -> xr.DataArray:
        """Convert item to image, return None if not possible."""
        _logger.debug(f"loading satpy item {item}")
        area = area or self._area

        scn = item.slot_files.scene
        scn.load([item.product])
        if area:
            scn = scn.resample(area)
        else:
            # composites of channels with different resolution would not work without resampling,
            # average the high-resolution channels
            # TODO: finest_area or coarsest_area? Note also the satpy bug
            # https://github.com/pytroll/satpy/issues/1595
            scn = scn.resample(scn.finest_area(), resampler="native")

        product = scn[item.product]
        da = get_enhanced_image(product).data
        da = da.transpose("bands", "y", "x")
        # TODO: make this optional, have a switch: all RGB, RGBA, BW, BWA, any
        if len(da.bands) == 4:
            # RGBA -> RGB
            da = da.isel(bands=slice(0, 3))
        elif len(da.bands) == 2:
            # LA -> L
            da = da.isel(bands=slice(0, 1))
            da = xr.concat([da] * 3, dim="bands")
        elif len(da.bands) == 1:
            # L -> RGB
            da = xr.concat([da] * 3, dim="bands")

        lon, lat = product.attrs["area"].get_lonlats()
        da.coords["lon"] = (("y", "x"), lon)
        da.coords["lat"] = (("y", "x"), lat)

        return da.persist()  # load data into memory

    def _find_items(
        self, base_path: Optional[Path] = None, slot_definition: Optional[SlotDefinition] = None
    ) -> List[SatpyProductFiles]:
        base_path = base_path or self._base_path
        slot_definition = slot_definition or self._slot_definition

        if not base_path.exists():
            raise ValueError(f"base folder {base_path} does not exist.")

        # find all available slots
        segment_gatherer = SegmentGatherer(slot_definition)
        slots = segment_gatherer.gather(base_path).values()

        # for each slot find all available products
        slot_products = []
        for slot in slots:
            for composite in slot.scene.available_composite_names():
                if self._area is None and composite in self._forbidden_composites_wo_area:
                    continue
                product_files = SatpyProductFiles(slot_files=slot, product=composite)
                slot_products.append(product_files)

        return slot_products

    def _extract_attrs(self, item: SatpyProductFiles) -> Dict[str, Any]:
        return dict(**item.slot_files.attrs, product=item.product)

    def _item2key(self, item: SatpyProductFiles) -> str:
        return f"{item.product}|{item.slot_files.key}"

    def _get_data(self, key: str) -> Optional[xr.DataArray]:
        """Return data given key, return None if not possible."""
        try:
            da = self._get_image(self._key2item(key))
        except KeyError:
            return None
        da.attrs.update(self._extract_attrs(self._key2item(key)))

        return da
