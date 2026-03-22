from __future__ import annotations


class RAGAppError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class EmptyQueryError(RAGAppError):
    def __init__(self, message: str = "Query must not be empty."):
        super().__init__(
            code="EMPTY_QUERY",
            message=message,
            status_code=400,
        )


class RetrievalError(RAGAppError):
    def __init__(self, message: str = "Failed to retrieve supporting context."):
        super().__init__(
            code="RETRIEVAL_FAILED",
            message=message,
            status_code=503,
        )


class GenerationError(RAGAppError):
    def __init__(
        self,
        message: str = "Failed to generate answer.",
        status_code: int = 502,
        code: str = "GENERATION_FAILED",
    ):
        super().__init__(
            code=code,
            message=message,
            status_code=status_code,
        )