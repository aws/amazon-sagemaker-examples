"""Provides named imports for lambda deployment."""

from api_batch_create.main import lambda_handler as api_batch_create_lambda_handler
from api_batch_metadata_post.main import lambda_handler as api_batch_post_metadata_lambda_handler
from api_batch_show.main import lambda_handler as api_batch_show_lambda_handler
from api_workforce_show.main import lambda_handler as api_workforce_show_lambda_handler
from labeling_job_state_change.main import lambda_handler as labeling_job_state_change_handler
from step_functions_batch_error.main import (
    lambda_handler as step_functions_batch_error_lambda_handler,
)
from step_functions_copy_logs_and_send_batch_completed.main import (
    lambda_handler as step_functions_copy_logs_and_send_batch_completed_lambda_handler,
)
from step_functions_send_second_level_sns_and_check_response.main import (
    lambda_handler as step_functions_send_second_level_sns_and_check_response_lambda_handler,
)
from step_functions_transformation.main import (
    lambda_handler as step_functions_transformation_lambda_handler,
)
from step_functions_trigger_labeling_job.main import (
    lambda_handler as step_functions_trigger_labeling_job_lambda_handler,
)
from step_functions_wait_batch_completion.main import (
    lambda_handler as step_functions_wait_batch_completion_lambda_handler,
)
from step_functions_wait_for_metadata_supply.main import (
    lambda_handler as step_functions_wait_batch_metadata_input_lambda_handler,
)
