"""Automated checks for call_cost(). Run with:  pytest

Each expected value was worked out by hand from the price sheet,
so these tests check the code against arithmetic we understand.
"""
import pytest

from lcvr.cost import call_cost

def make_record(input_tokens=0, cache_write=0, cache_read=0, output=0, ok=True):
    """Build a fake log line with just the fields call_cost() needs."""
    return {
        "ok": ok,
        "input_tokens": input_tokens,
        "cache_write_tokens": cache_write,
        "cache_read_tokens": cache_read,
        "output_tokens": output,
    }


def test_cache_write_call():
    # 12 x 0.10 + 6101 x 0.125 + 17 x 0.50, all per million
    record = make_record(input_tokens=12, cache_write=6101, output=17)
    assert call_cost(record) == pytest.approx(0.000772325)


def test_cache_read_call():
    # 12 x 0.10 + 6101 x 0.01 + 16 x 0.50, all per million
    record = make_record(input_tokens=12, cache_read=6101, output=16)
    assert call_cost(record) == pytest.approx(0.00007021)


def test_failed_attempt_costs_nothing():
    record = make_record(ok=False)
    assert call_cost(record) == 0.0


def test_exactly_100000_is_still_short():
    # 100,000 input tokens x $0.10 per million = $0.01
    record = make_record(input_tokens=100_000)
    assert call_cost(record) == pytest.approx(0.01)


def test_one_token_over_is_long():
    # 100,001 input tokens x $0.50 per million = $0.0500005
    record = make_record(input_tokens=100_001)
    assert call_cost(record) == pytest.approx(0.0500005)


def test_cache_reads_count_toward_the_threshold():
    # 20 + 99,990 = 100,010 tokens -> long tier for EVERYTHING, output included
    # 20 x 0.50 + 99,990 x 0.05 + 10 x 2.50, all per million
    record = make_record(input_tokens=20, cache_read=99_990, output=10)
    assert call_cost(record) == pytest.approx(0.0050345)


def test_long_context_512k_cache_read():
    # 30 x 0.50 + 512,000 x 0.05 + 10 x 2.50, all per million
    record = make_record(input_tokens=30, cache_read=512_000, output=10)
    assert call_cost(record) == pytest.approx(0.02564)