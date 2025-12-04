import numpy as np
import pytest

from src import data_prep


def _sample_frame():
    return data_prep.make_fake_data(n_events=30, seed=123)


def test_pacing_median_is_monotonic():
    """
    The global pacing curve should be monotonically non-decreasing as we approach
    the event day (d_bin going from 120 → 0 means cum_pct should increase).
    """
    df = _sample_frame()
    pacing = data_prep.derive_core(df)["global_pacing_curve"]
    if pacing.empty:
        pytest.skip("No pacing data generated")
    
    # Sort by d_bin ascending to check monotonicity (0 → 120)
    pacing_sorted = pacing.sort_values("d_bin", ascending=True)
    median = pacing_sorted["median_cum_pct"].dropna().to_numpy()
    
    # As d_bin increases (going further from event), cum_pct should decrease (or stay flat)
    # So diff should be <= 0 (with small tolerance)
    diffs = np.diff(median)
    assert np.all(diffs <= 1e-6), f"Pacing curve not monotonic: {diffs}"


def test_category_shares_are_bounded_and_sum_to_one():
    df = _sample_frame()
    derived = data_prep.derive_core(df)

    for key in ("category_share_pre", "category_share_post"):
        share_df = derived[key]
        if share_df.empty:
            continue
        shares = share_df["share_ratio"]
        assert np.all((shares >= 0) & (shares <= 1))
        assert shares.sum() == pytest.approx(1, rel=1e-6, abs=1e-6)
