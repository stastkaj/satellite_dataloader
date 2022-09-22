from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import copy
from functools import partial
import logging
from pathlib import Path

from pyresample import AreaDefinition
from satpy.writers import get_enhanced_image
import xarray as xr

from satdl.datasets._dataset_base import AttributeDatasetBase
from satdl.datasets._segment_gatherer import SatpySlotFiles, SlotDefinition, SegmentGatherer


_logger = logging.getLogger(__name__)


class SatpyFolderDataset(AttributeDatasetBase[SatpySlotFiles, str, xr.DataArray]):
    _forbidden_composites_wo_area = [
        'cloud_phase_distinction',
        'cloud_phase_distinction_raw',
        'hrv_clouds',
        'hrv_fog',
        'hrv_severe_storms',
        'hrv_severe_storms_masked',
        'ir108_3d',
        'ir_sandwich',
        'natural_color_with_night_ir',
        'natural_color_with_night_ir_hires',
        'natural_enh_with_night_ir',
        'natural_enh_with_night_ir_hires',
        'night_ir_with_background',
        'night_ir_with_background_hires',
        'realistic_colors',
        'vis_sharpened_ir'
    ]  # TODO: find why it does not work, move the list to some config file
    def __init__(
        self,
        base_path: Union[str, Path],
        slot_definition: SlotDefinition,
        area: Optional[Union[str, AreaDefinition]]
        # max_cache: Optional[int] = None
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
        # max_cache: int, optional
        #     Maximum number of images that will be cached.
        """
        self._base_path = Path(base_path)
        self._slot_definition = slot_definition
        self._area = area

        super().__init__()

        #self._get_image = lru_cache(max_cache)(SatpyFolderDataset.data2image)
        self._get_image = partial(self.data2image, area=area)

    def data2image(self, slot: SatpySlotFiles, area: Optional[Union[str, AreaDefinition]]) -> xr.DataArray:
        """Convert item to image, return None if not possible."""
        area = area or self._area
        product = slot.attrs['product']

        scn = slot.scene
        scn.load([product])
        if area:
            scn = scn.resample(area)

        da = get_enhanced_image(scn[product]).data
        da = da.transpose('bands', 'y', 'x')
        # TODO: make this optional, have a switch: all RGB, RGBA, BW, BWA, any
        if len(da.bands) == 4:
            # RGBA -> RGB
            da = da.isel(bands=slice(0, 3))
        elif len(da.bands) == 2:
            # LA -> L
            da = da.isel(bands=slice(0, 1))
            da = xr.concat([da] * 3, dim='bands')
        elif len(da.bands) == 1:
            # L -> RGB
            da = xr.concat([da]*3, dim='bands')

        return da

    def _find_items(
        self, base_path: Optional[Path] = None, slot_definition: Optional[SlotDefinition] = None
    ) -> List[Path]:
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
                slot_ = copy.deepcopy(slot)
                slot_.attrs['product'] = composite
                if slot_._key is not None:
                    raise NotImplementedError('Explict key exists.')  # TODO: implement
                slot_products.append(slot_)

        return slot_products

    def _extract_attrs(self, item: Path) -> Dict[str, Any]:
        return item.attrs

    def _item2key(self, item: Path) -> str:
        return item.key

    def _get_data(self, key: str) -> xr.DataArray:
        """Return data given key, return None if not possible."""
        try:
            da = self._get_image(self._key2item(key))
        except KeyError:
            return None
        da.attrs.update(self._extract_attrs(self._key2item(key)))

        return da
