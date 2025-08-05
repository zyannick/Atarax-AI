class AtaraxAIError(Exception):
    pass


class CoreAIServiceError(AtaraxAIError):
    pass


class ServiceInitializationError(AtaraxAIError):
    pass


class ValidationError(AtaraxAIError):
    pass


class AtaraxAIAuthenticationError(AtaraxAIError):
    pass


class AtaraxAIStateError(AtaraxAIError):
    pass


class AtaraxAILockError(AtaraxAIError):
    pass
