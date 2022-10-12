import time
from sagemaker.estimator import EstimatorBase

def fit_with_retries(retries : int, estimator : EstimatorBase, *args, **kwargs):
    """Run estimator fit with retries in case of temporary issues like capacity exception or user exceeded resource usage
    Example invocation: fit_with_retries(5, estimator, job_name="my job name")

    Args:
        retries (int): How many retries in case of exception_to_try is raised
        estimator (EstimatorBase): will call estimator.fit(...)
        *args: list of positioned arguments to pass to fit()
        **kwargs: list of keyword arguments to pass to fit()
    Returns:
        None
    """
    orig_job_name = kwargs['job_name'] if 'job_name' in kwargs and kwargs['job_name'] else None
    for i in range(1, retries+1):
        try:
            # Ensure job_name is unique between retries (if specified)
            if orig_job_name:
                 kwargs['job_name'] = orig_job_name + f'-{i}'
            estimator.fit(*args, **kwargs)
            break
        except Exception as e:
            if not ('CapacityError' in str(e) or 'ResourceLimitExceeded' in str(e)):
                raise e
            print(f'Caught error: {type(e).__name__}: {e}')
            if i == retries:
                print(f'Giving up after {retries} failed attempts.')
                raise e
            else:
                if 'ResourceLimitExceeded' in str(e):
                    seconds = 10
                    print(f'ResourceLimitExceeded: Sleeping {seconds}s before retrying.')
                    time.sleep(seconds)
                print(f'Retrying attempt: {i+1}/{retries}')
                continue