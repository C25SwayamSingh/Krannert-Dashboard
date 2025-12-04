from src import utils


def test_categorize_event_mappings():
    cases = {
        "Musical": "Theatre",
        "opera": "Theatre",
        "Festival": "Festivals",
        "Talk": "Talks",
        "concert": "Concerts",
        "Unknown": "Unknown",
    }
    for raw, expected in cases.items():
        assert utils.categorize_event(raw) == expected


def test_categorize_event_fallback_uses_secondary_value():
    assert utils.categorize_event(None, "concert") == "Concerts"

