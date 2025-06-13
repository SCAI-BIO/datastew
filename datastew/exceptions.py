from typing import Optional


class ObjectStorageError(Exception):
    """Raised when stroing an object in the repository fails"""

    def __init__(self, message: str, original_exception: Optional[Exception]):
        super().__init__(message)
        self.original_exception = original_exception
