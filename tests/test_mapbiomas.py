from agro_predictor.mapbiomas import (
    CLASS_MAP,
    _accept_window_points,
    _selection_pool,
    _should_stop,
)


def _sample_pools():
    return {"Pasture": []}, {"Pasture": []}, {"Pasture": set()}


def test_accept_window_points_caps_primary_and_records_overflow_provenance():
    primary, overflow, primary_windows = _sample_pools()
    points = [(float(index), -20.0) for index in range(6)]

    _accept_window_points(primary, overflow, primary_windows, "Pasture", points, 7, 2)

    assert primary["Pasture"] == [(0.0, -20.0, 7), (1.0, -20.0, 7)]
    assert overflow["Pasture"] == [
        (2.0, -20.0, 7),
        (3.0, -20.0, 7),
    ]
    retained = primary["Pasture"] + overflow["Pasture"]
    assert (4.0, -20.0, 7) not in retained
    assert (5.0, -20.0, 7) not in retained


def test_selection_pool_does_not_use_overflow_when_primary_is_full():
    primary = [(0.0, 0.0, index) for index in range(5)]
    overflow = [(1.0, 1.0, 10)]

    pool, used_overflow = _selection_pool(primary, overflow, per_class=5)

    assert pool == primary
    assert used_overflow == []


def test_selection_pool_top_up_is_deterministic_by_window_then_insertion_order():
    primary = [(0.0, 0.0, 0), (0.1, 0.1, 0)]
    overflow = [
        (3.0, 3.0, 3),
        (1.0, 1.0, 1),
        (1.1, 1.1, 1),
        (2.0, 2.0, 2),
    ]

    first = _selection_pool(primary, overflow, per_class=5)
    second = _selection_pool(primary, overflow, per_class=5)

    assert first == second
    assert first[1] == [
        (1.0, 1.0, 1),
        (1.1, 1.1, 1),
        (2.0, 2.0, 2),
    ]
    assert first[0] == [*primary, *first[1]]


def test_accept_window_points_caps_each_pool_per_window_without_overall_limit():
    primary, overflow, primary_windows = _sample_pools()
    per_window_cap = 2
    window_count = 20

    for window_index in range(window_count):
        points = [(float(index), float(window_index)) for index in range(10)]
        primary_before = len(primary["Pasture"])
        overflow_before = len(overflow["Pasture"])
        _accept_window_points(
            primary,
            overflow,
            primary_windows,
            "Pasture",
            points,
            window_index,
            per_window_cap,
        )
        assert len(primary["Pasture"]) - primary_before <= per_window_cap
        assert len(overflow["Pasture"]) - overflow_before <= per_window_cap

    assert len(primary["Pasture"]) == window_count * per_window_cap
    assert len(overflow["Pasture"]) == window_count * per_window_cap


def test_accept_window_points_tracks_primary_windows_not_overflow_points():
    primary, overflow, primary_windows = _sample_pools()

    _accept_window_points(
        primary,
        overflow,
        primary_windows,
        "Pasture",
        [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)],
        4,
        1,
    )
    _accept_window_points(
        primary,
        overflow,
        primary_windows,
        "Pasture",
        [(1.0, 1.0), (1.1, 1.1)],
        9,
        1,
    )
    _accept_window_points(
        primary,
        overflow,
        primary_windows,
        "Pasture",
        [(2.0, 2.0)],
        12,
        1,
    )

    assert primary_windows["Pasture"] == {4, 9, 12}
    assert {point[2] for point in overflow["Pasture"]} == {4, 9}


def test_class_ids_with_same_label_share_pools_and_caps():
    class_12_label = CLASS_MAP[12]
    class_15_label = CLASS_MAP[15]
    assert class_12_label == class_15_label == "Pasture"

    primary = {class_12_label: []}
    overflow = {class_12_label: []}
    primary_windows = {class_12_label: set()}

    window_points = {class_12_label: []}
    window_points[class_12_label].extend(
        [(12.0, 0.0), (12.1, 0.0), (12.2, 0.0)]
    )
    window_points[class_15_label].extend([(15.0, 0.0), (15.1, 0.0)])
    _accept_window_points(
        primary,
        overflow,
        primary_windows,
        class_12_label,
        window_points["Pasture"],
        7,
        2,
    )

    assert primary["Pasture"] == [(12.0, 0.0, 7), (12.1, 0.0, 7)]
    assert overflow["Pasture"] == [(12.2, 0.0, 7), (15.0, 0.0, 7)]
    assert (15.1, 0.0, 7) not in primary["Pasture"] + overflow["Pasture"]
    assert primary_windows["Pasture"] == {7}


def test_accept_window_points_keeps_collecting_after_reaching_per_class():
    per_class = 5
    primary = {"Pasture": [(float(index), 0.0, index) for index in range(per_class)]}
    overflow = {"Pasture": []}
    primary_windows = {"Pasture": set(range(per_class))}

    _accept_window_points(
        primary,
        overflow,
        primary_windows,
        "Pasture",
        [(10.0, 1.0), (11.0, 1.0), (12.0, 1.0), (13.0, 1.0)],
        99,
        2,
    )

    assert len(primary["Pasture"]) == per_class + 2
    assert primary["Pasture"][-2:] == [(10.0, 1.0, 99), (11.0, 1.0, 99)]
    assert overflow["Pasture"] == [(12.0, 1.0, 99), (13.0, 1.0, 99)]
    assert 99 in primary_windows["Pasture"]


def test_should_stop_only_after_pools_are_full_and_minimum_windows_visited():
    not_full = {"Forest": [(0.0, 0.0, 0)], "Pasture": []}
    full = {"Forest": [(0.0, 0.0, 0)], "Pasture": [(1.0, 1.0, 0)]}

    assert not _should_stop(not_full, per_class=1, windows_visited=100, min_windows=40)
    assert not _should_stop(full, per_class=1, windows_visited=39, min_windows=40)
    assert _should_stop(full, per_class=1, windows_visited=40, min_windows=40)
