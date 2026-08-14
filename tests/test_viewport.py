from context_scythe.environment.viewport import compute_viewport_bids, filter_axtree_by_bids


def test_visible_elements_are_kept():
    props = {
        "1": {"visibility": 1.0},
        "2": {"visibility": 0.5},
        "3": {"visibility": 0.75},
    }

    assert compute_viewport_bids(props) == {"1", "2", "3"}


def test_partially_and_fully_hidden_elements_are_dropped():
    props = {
        "1": {"visibility": 1.0},
        "2": {"visibility": 0.49},
        "3": {"visibility": 0.0},
    }

    assert compute_viewport_bids(props) == {"1"}


def test_elements_without_visibility_are_dropped():
    props = {
        "1": {"visibility": 1.0},
        "2": {"bbox": [100, 88, 50, 20]},
        "3": {},
        "4": {"visibility": None},
    }

    assert compute_viewport_bids(props) == {"1"}


def test_bids_are_returned_as_strings():
    props = {
        171: {"visibility": 1.0},
        "1275": {"visibility": 1.0},
    }

    assert compute_viewport_bids(props) == {"171", "1275"}


def test_empty_properties_yield_no_bids():
    assert compute_viewport_bids({}) == set()


def test_filter_keeps_visible_elements_from_saved_rollout_shape():
    axtree = (
        "RootWebArea 'Orders'\n"
        "\t[171] link 'Magento Admin Panel', center=\"(44,38)\", clickable, visible\n"
        "\t[1275] textbox 'per page', center=\"(1043,98)\", clickable, visible\n"
        "\t[4009] link 'Report an Issue', center=\"(1182,910)\", clickable, visible\n"
        "\t[5001] link 'Offscreen Footer Link', center=\"(1182,4910)\", clickable\n"
    )
    props = {
        "171": {"bbox": [0.0, 0.0, 88.0, 75.0], "visibility": 1.0},
        "1275": {"bbox": [1013.0, 88.0, 60.0, 21.0], "visibility": 1.0},
        "4009": {"bbox": [1130.0, 900.484375, 105.0, 19.0], "visibility": 1.0},
        "5001": {"bbox": [1130.0, 4900.0, 105.0, 19.0], "visibility": 0.0},
    }

    bids = compute_viewport_bids(props)
    filtered = filter_axtree_by_bids(axtree, bids)

    assert "Magento Admin Panel" in filtered
    assert "per page" in filtered
    assert "Report an Issue" in filtered
    assert "Offscreen Footer Link" not in filtered
    # The ancestor line is preserved so the tree stays well-formed.
    assert filtered.startswith("RootWebArea 'Orders'")


def test_filter_keeps_static_text_children_of_visible_elements():
    axtree = (
        "RootWebArea 'Orders'\n"
        "\t[171] link 'Magento Admin Panel'\n"
        "\t\tStaticText 'Magento'\n"
        "\t\t[172] button 'Nested button'\n"
        "\t[900] link 'Hidden link'\n"
        "\t\tStaticText 'Hidden text'\n"
    )
    props = {
        "171": {"visibility": 1.0},
        "172": {"visibility": 0.0},
        "900": {"visibility": 0.0},
    }

    filtered = filter_axtree_by_bids(axtree, compute_viewport_bids(props))

    assert "StaticText 'Magento'" in filtered
    assert "Nested button" not in filtered
    assert "Hidden link" not in filtered
    assert "Hidden text" not in filtered
