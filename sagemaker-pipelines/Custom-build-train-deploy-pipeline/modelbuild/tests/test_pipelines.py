import pytest


@pytest.mark.xfail
def test_that_you_wrote_tests():
    assert False, "No tests written"


def test_pipelines_importable():
    import pipelines  # noqa: F401
