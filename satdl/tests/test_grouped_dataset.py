import pytest

from satdl.datasets import GroupedDataset


@pytest.fixture
def grouped_dataset() -> GroupedDataset:
    dataset = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}

    return GroupedDataset(
        dataset,
        key_groups=[["a", "b"], ["c", "e", "d"], ["f"]],
        group_attrs=[{"attr1": 1}, {"attr1": 2, "attr2": 1}, {"attr1": 2, "attr2": 2}],
    )


def test_grouped_dataset_properties(grouped_dataset: GroupedDataset) -> None:
    # has correct length
    assert len(grouped_dataset) == 3

    # returns correct group attributes
    assert grouped_dataset.attrs == [{"attr1": 1}, {"attr1": 2, "attr2": 1}, {"attr1": 2, "attr2": 2}]

    # returns correct groups
    for group, ground_truth in zip(grouped_dataset, [[1, 2], [3, 5, 4], [6]]):
        assert list(group) == ground_truth

    # __getitem__ works
    assert list(grouped_dataset[2]) == [6]


def test_grouped_dataset_filter(grouped_dataset: GroupedDataset) -> None:
    # forbidden groups - one existing, one non-existing
    filtered_ds = grouped_dataset.filter(
        forbidden_groups=[{"attr1": 2, "attr2": 1}, {"attr1": 1, "attr2": 1}]
    )

    assert isinstance(filtered_ds, GroupedDataset)
    assert len(filtered_ds) == 2
    assert filtered_ds.attrs == [{"attr1": 1}, {"attr1": 2, "attr2": 2}]
    for group, ground_truth in zip(filtered_ds, [[1, 2], [6]]):
        assert list(group) == ground_truth

    # requested_groups - one existing, one non-existing
    filtered_ds = grouped_dataset.filter(
        requested_groups=[{"attr1": 2, "attr2": 1}, {"attr1": 1, "attr2": 1}]
    )

    assert len(filtered_ds) == 1
    assert filtered_ds.attrs == [{"attr1": 2, "attr2": 1}]
    for group, ground_truth in zip(filtered_ds, [[3, 5, 4]]):
        assert list(group) == ground_truth

    # forbidden attributes - one existing value one non-existing
    filtered_ds = grouped_dataset.filter(forbidden_attrs={"attr2": [1, 2], "attr3": 1})

    assert len(filtered_ds) == 1
    assert filtered_ds.attrs == [{"attr1": 1}]
    for group, ground_truth in zip(filtered_ds, [[1, 2]]):
        assert list(group) == ground_truth

    # requested attributes - one existing value one non-existing
    filtered_ds = grouped_dataset.filter(requested_attrs={"attr2": [1, 2, 3], "attr3": 1})

    assert len(filtered_ds) == 0

    # requested attributes - both existing
    filtered_ds = grouped_dataset.filter(requested_attrs={"attr2": [1, 2], "attr1": [1, 2]})

    assert len(filtered_ds) == 2
    assert filtered_ds.attrs == [{"attr1": 2, "attr2": 1}, {"attr1": 2, "attr2": 2}]
    for group, ground_truth in zip(filtered_ds, [[3, 5, 4], [6]]):
        assert list(group) == ground_truth

    # no filter
    assert grouped_dataset.filter() is grouped_dataset
