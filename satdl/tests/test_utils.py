from satdl.utils import tolist


def test_tolist_conversion() -> None:
    assert tolist("3", converter=int) == [3]
