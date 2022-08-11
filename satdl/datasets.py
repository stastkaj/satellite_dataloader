from typing import Any, Callable, Dict, Generator, Iterable, KeysView, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
from trollsift import Parser
import xarray as xr

from .utils import image2xr


_logger = logging.getLogger(__name__)


def _get_get_image(
    georef: Optional[Union[str, Path, xr.DataArray]]
) -> Callable[[Union[str, Path]], xr.DataArray]:
    def _get_image(path: Union[str, Path]) -> xr.DataArray:
        _logger.debug(f"loading georeferenced image {path}")
        return image2xr(path, georef=georef).load()

    return _get_image


class StaticImageFolderDataset:
    def __init__(
        self,
        base_folder: Union[str, Path],
        file_mask: Union[str, Parser],
        georef: Optional[Union[str, Path, xr.DataArray]] = None,
        max_cache: Optional[int] = 0,
    ) -> None:
        """Create ImageFolderDataset

        Note: content of the folder is scanned only once, at the class creation

        Parameters
        ----------
        base_folder : str or Path
            root folder of the data
        file_mask : str or trollsift.Parser
            mask of image names specifying attributes in the file name. Must not contain wildcards '*' or '?',
            should be relatie to base_folder
        georef : str or Path or xr.DataArray or None
            external georeference for plain images, optional
        max_cache: int, optional
            Maximum number of images that will be cached.
        """
        self._base_folder = Path(base_folder)
        if not self._base_folder.exists():
            raise ValueError(f"base folder {base_folder} does not exist")
        self._file_mask = Parser(file_mask)
        self._georef = georef  # TODO: validate georeference
        self._files = list(self._base_folder.rglob(self._file_mask.globify()))
        self._attrs = {self._filename2key(f): self._extract_attrs(f, relative=False) for f in self._files}

        self._get_image = lru_cache(max_cache)(_get_get_image(self._georef))

    def __len__(self) -> int:
        return len(self._files)

    def _extract_attrs(self, filename: Union[str, Path], relative: bool = False) -> Dict[str, Any]:
        key = str(filename) if relative else self._filename2key(filename)
        return self._file_mask.parse(key)

    def _filename2key(self, filename: Union[str, Path]) -> str:
        return str(Path(filename).relative_to(self._base_folder))

    def keys(self) -> KeysView[str]:
        """Return all image keys"""
        return self._attrs.keys()

    @property
    def attrs(self) -> Dict[str, Dict[str, Any]]:
        """Return dict {key: attrs_dict}"""
        return self._attrs

    def items(self) -> List[Tuple[str, Any]]:
        """Return list of (key, attributes) pairs"""
        return [(key, self[key]) for key in self.keys()]

    def __getitem__(self, key: str) -> xr.DataArray:
        """Return image as DataArray from key"""
        da = self._get_image(self._base_folder / key)
        da.attrs.update(self._extract_attrs(key, relative=True))

        return da

    def iloc(self, i: int) -> xr.DataArray:
        """Return i-th image as DataArrray

        Raises IndexError if i >= len(self)
        """
        return self[self._filename2key(self._files[i])]

    def random(self) -> xr.DataArray:
        """Return random image as DataArray"""
        return self.iloc(np.random.randint(len(self)))

    def __iter__(self) -> Generator[str, None, None]:
        return (key for key in self.keys())

    def groupby(
        self, attr_name: str, sortby: Optional[Union[str, List[str]]] = None, ascending: bool = True
    ) -> "GroupedDataset":
        sortby = sortby or []
        if isinstance(sortby, str):
            sortby = [sortby]
        groups = defaultdict(lambda: [])
        for key, key_attrs in sorted(
            self.attrs.items(),
            key=lambda x: tuple(x[1][sort_col] for sort_col in sortby),  # type: ignore
            reverse=not ascending,
        ):
            groups[key_attrs.get(attr_name)].append(key)

        return GroupedDataset(
            self, key_groups=groups.values(), shared_attrs=tuple({attr_name: k} for k in groups.keys())
        )


class GroupedDataset:
    def __init__(
        self,
        parent: StaticImageFolderDataset,
        key_groups: Iterable[List[str]],
        shared_attrs: Sequence[Dict[str, Any]],
    ):  # TODO: StaticImageFolderDataset -> some more generic type
        """Dataset representing items from a parent dataset grouped by some condition.

        Usage:
            grouped_dataset[0] ... returns generator of all elements of i-th group
            for group in grouped_dataset:
                # group ~ grouped_dataset[i], i.e. it is a generator of all elemenents in the i-th group

        Parameters
        ----------
        parent: The parent dataset.
        key_groups: Iterable of lists of parent keys. Each list represents one group and contains
            all keys that belong to that group.
        shared_attrs: Enumeration of attributes that defined the groups.
            E.g. [{product: 'IR108', date: '20201018}, {product: 'HRV', date: '20201018'},
                  {product: 'IR108', data: '20201019}]
        """
        self._parent = parent
        self._key_groups = list(key_groups)
        self._shared_attrs = list(shared_attrs)

        if len(self._key_groups) != len(self._shared_attrs):
            raise ValueError(
                f"len(key_groups) != len(shared_attrs): "
                f"{len(self._key_groups)} != {len(self._shared_attrs)}"
            )

    def __len__(self) -> int:
        return len(self._key_groups)

    def __getitem__(self, i: int) -> Generator[xr.DataArray, Any, Any]:
        """Get all elements of i-th group"""
        return (self._parent[key] for key in self._key_groups[i])

    @property
    def attrs(self) -> List[Dict[str, Any]]:
        return self._shared_attrs

    def __iter__(self) -> Generator[Generator[xr.DataArray, None, None], None, None]:
        return (self[i] for i in range(len(self)))
