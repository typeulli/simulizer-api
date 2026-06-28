"""
Wire protocol between Server and Worker.

The wrapper messages carry an opaque `payload` (dict) so that the library
itself stays generic over T (server -> worker) and U (worker -> server).
Server/Worker convert payload <-> T/U using their own pydantic types.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class WorkerHello(BaseModel):
    """First message a worker sends after connecting."""
    priority: int  # lower number = higher priority


class ServerToWorker(BaseModel):
    request_id: str
    type: Literal["request", "cancel"]
    payload: Optional[dict[str, Any]] = None  # T as dict; None for cancel


class WorkerToServer(BaseModel):
    request_id: str
    payload: dict[str, Any]  # U as dict
    is_final: bool
