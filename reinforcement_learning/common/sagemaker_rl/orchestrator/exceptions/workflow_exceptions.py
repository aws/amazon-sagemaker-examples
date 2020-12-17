class UnhandledWorkflowException(Exception):
    pass

class SageMakerTrainingJobException(Exception):
    pass

class SageMakerHostingException(Exception):
    pass

class WorkflowJoiningJobException(Exception):
    pass

class EvalScoreNotAvailableException(Exception):
    pass

class JoinQueryIdsNotAvailableException(Exception):
    pass

class InvalidUsageException(Exception):
    pass