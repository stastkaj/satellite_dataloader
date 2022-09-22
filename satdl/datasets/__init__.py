from .gridded_dataset import GriddedDataset
from .image_folder import ImageFolderDataset
from .satpy_folder import SatpyFolderDataset
from ._segment_gatherer import SlotDefinition


__all__ = ("GriddedDataset", "ImageFolderDataset", "SatpyFolderDataset", "SlotDefinition")
