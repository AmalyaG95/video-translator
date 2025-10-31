from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TranslationRequest(_message.Message):
    __slots__ = ("session_id", "file_path", "source_lang", "target_lang")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LANG_FIELD_NUMBER: _ClassVar[int]
    TARGET_LANG_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    file_path: str
    source_lang: str
    target_lang: str
    def __init__(self, session_id: _Optional[str] = ..., file_path: _Optional[str] = ..., source_lang: _Optional[str] = ..., target_lang: _Optional[str] = ...) -> None: ...

class TranslationProgress(_message.Message):
    __slots__ = ("session_id", "progress", "current_step", "status", "message", "early_preview_available", "early_preview_path")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STEP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    EARLY_PREVIEW_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    EARLY_PREVIEW_PATH_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    progress: float
    current_step: str
    status: str
    message: str
    early_preview_available: bool
    early_preview_path: str
    def __init__(self, session_id: _Optional[str] = ..., progress: _Optional[float] = ..., current_step: _Optional[str] = ..., status: _Optional[str] = ..., message: _Optional[str] = ..., early_preview_available: bool = ..., early_preview_path: _Optional[str] = ...) -> None: ...

class ResultRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class TranslationResult(_message.Message):
    __slots__ = ("session_id", "output_path", "status", "duration", "original_srt", "translated_srt")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_SRT_FIELD_NUMBER: _ClassVar[int]
    TRANSLATED_SRT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    output_path: str
    status: str
    duration: float
    original_srt: str
    translated_srt: str
    def __init__(self, session_id: _Optional[str] = ..., output_path: _Optional[str] = ..., status: _Optional[str] = ..., duration: _Optional[float] = ..., original_srt: _Optional[str] = ..., translated_srt: _Optional[str] = ...) -> None: ...

class CancelRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class CancelResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
