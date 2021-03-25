class RecordAlreadyExistsException(Exception):
    pass

class ConcurrentModificationException(Exception):
    pass

class ConditionalCheckFailure(Exception):
    pass