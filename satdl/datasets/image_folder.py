from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
from functools import lru_cache
import logging
from pathlib import Path

from trollsift import Parser
import xarray as xr

from satdl.datasets._dataset_base import AttributeDatasetBase
from satdl.utils import image2xr


_logger = logging.getLogger(__name__)


def _get_get_image(
    georef: Optional[Union[str, Path, xr.DataArray]]
) -> Callable[[Union[str, Path]], xr.DataArray]:
    def _get_image(path: Union[str, Path]) -> xr.DataArray:
        _logger.debug(f"loading georeferenced image {path}")
        return image2xr(path, georef=georef).load()

    return _get_image


class ImageFolderDataset(AttributeDatasetBase[Path, str, xr.DataArray]):
    def __init__(
        self,
        base_path: Union[str, Path],
        file_mask: Union[str, Parser],
        georef: Optional[Union[str, Path, xr.DataArray]] = None,
        max_cache: Optional[int] = 0,
    ) -> None:
        """Dataset of georeferenced images in a local folder.

        Note: content of the folder is scanned only once, at the class creation

        Parameters
        ----------
        base_path : str or Path
            root folder of the data
        file_mask : str or trollsift.Parser
            mask of image names specifying attributes in the file name. Must not contain wildcards '*' or '?',
            should be relatie to base_path
        georef : str or Path or xr.DataArray or None
            external georeference for plain images, optional
        max_cache: int, optional
            Maximum number of images that will be cached.
        """
        self._base_path = Path(base_path)
        self._file_mask = Parser(file_mask) if not isinstance(file_mask, Parser) else file_mask
        self._georef = georef  # TODO: validate georeference
        self._relative_key = True

        super().__init__()

        self._get_image = lru_cache(max_cache)(_get_get_image(self._georef))

    def _find_items(self, base_path: Optional[Path] = None, file_mask: Optional[Parser] = None) -> List[Path]:
        base_path = base_path or self._base_path
        file_mask = file_mask or self._file_mask

        if not base_path.exists():
            raise ValueError(f"base folder {base_path} does not exist.")

        return list(base_path.rglob(file_mask.globify()))

    def _extract_attrs(self, item: Path, relative: Optional[bool] = None) -> Dict[str, Any]:
        if relative is None:
            relative = self._relative_key
        key = self._item2key(item, relative=relative)
        return self._file_mask.parse(key)

    def _item2key(self, item: Path, relative: Optional[bool] = None) -> str:
        if relative is None:
            relative = self._relative_key
        return str(item.relative_to(self._base_path)) if relative else str(item)

    def _get_data(self, key: str) -> xr.DataArray:
        """Return data given key."""
        da = self._get_image(self._key2item(key))
        da.attrs.update(self._extract_attrs(self._key2item(key), relative=True))

        return da
