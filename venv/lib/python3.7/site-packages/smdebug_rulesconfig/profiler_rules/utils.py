def validate_positive_integer(key, val):
    assert val > 0, f"{key} must be a positive integer!"


def validate_percentile(key, val):
    assert 0 <= val <= 100, f"{key} must be a valid percentile!"
