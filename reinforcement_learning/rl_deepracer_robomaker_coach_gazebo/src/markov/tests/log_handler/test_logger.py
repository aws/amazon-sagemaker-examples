""" The script takes care of testing the functionality of logger.py
"""
import pytest
import logging
from markov.log_handler.logger import Logger

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_get_logger():
    """The test function checks if the Logger is instantiated properly.
    """
    log = Logger(__name__, logging.INFO).get_logger()
    assert isinstance(log, logging.Logger)
