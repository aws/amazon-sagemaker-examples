""" The script takes care of testing the functionality of test_deepracer_exceptions.py
"""
import pytest
from markov.log_handler.deepracer_exceptions import (RewardFunctionError, GenericRolloutException,
                                                     GenericTrainerException, GenericTrainerError,
                                                     GenericRolloutError, GenericValidatorException,
                                                     GenericValidatorError, GenericException,
                                                     GenericError)

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_deepracer_exceptions():
    """The function tests whether the user defined exceptions in deepracer_exceptions.py are
    getting raised properly when we call them from any part of SIMAPP code.

    The test function also checks whether the superclass Exception manages to provide
    the necessary error message passed along as well.

    Raises:
        RewardFunctionError
        GenericTrainerException
        GenericTrainerError
        GenericRolloutException
        GenericRolloutError
        GenericValidatorException
        GenericValidatorError
        GenericException
        GenericError
    """
    with pytest.raises(RewardFunctionError, match=r".*RewardFunctionError.*"):
        raise RewardFunctionError("RewardFunctionError")
    with pytest.raises(GenericTrainerException, match=r".*GenericTrainerException.*"):
        raise GenericTrainerException("GenericTrainerException")
    with pytest.raises(GenericTrainerError, match=r".*GenericTrainerError.*"):
        raise GenericTrainerError("GenericTrainerError")
    with pytest.raises(GenericRolloutException, match=r".*GenericRolloutException.*"):
        raise GenericRolloutException("GenericRolloutException")
    with pytest.raises(GenericRolloutError, match=r".*GenericRolloutError.*"):
        raise GenericRolloutError("GenericRolloutError")
    with pytest.raises(GenericValidatorException, match=r".*GenericValidatorException.*"):
        raise GenericValidatorException("GenericValidatorException")
    with pytest.raises(GenericValidatorError, match=r".*GenericValidatorError.*"):
        raise GenericValidatorError("GenericValidatorError")
    with pytest.raises(GenericException, match=r".*GenericException.*"):
        raise GenericException("GenericException")
    with pytest.raises(GenericError, match=r".*GenericError.*"):
        raise GenericError("GenericError")
