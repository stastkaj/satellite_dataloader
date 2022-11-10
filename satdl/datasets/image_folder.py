from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
from functools import partial
import logging
from pathlib import Path

from trollsift import Parser
import xarray as xr

from satdl.datasets._dataset_base import AttributeDatasetBase
from satdl.utils import image2xr


_logger = logging.getLogger(__name__)


class ImageFolderDataset(AttributeDatasetBase[Path, str, xr.DataArray]):
    def __init__(
        self,
        base_path: Union[str, Path],
        file_mask: Union[str, Parser],
        georef: Optional[Union[str, Path, xr.DataArray]] = None,
        cache: Optional[Callable] = None,
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
        cache: Callable
            Cache decorator used to cache dataset outputs.
        """
        self._base_path = Path(base_path)
        self._file_mask = Parser(file_mask) if not isinstance(file_mask, Parser) else file_mask
        self._georef = georef  # TODO: validate georeference
        self._relative_key = True

        super().__init__()

        _get_image = partial(self._data2image, georef=self._georef)
        self._get_image = cache(_get_image) if cache is not None else _get_image

    def _data2image(
        self, path: Union[str, Path], georef: Optional[Union[str, Path, xr.DataArray]]
    ) -> xr.DataArray:
        _logger.debug(f"loading georeferenced image {path}")
        return image2xr(path, georef=georef).load()

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
