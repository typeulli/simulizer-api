"""
Generic Worker[T, U] that connects to a Server over WebSocket, receives
typed requests of type T, and streams back typed responses of type U.

Usage:
    async def handle(req: MyRequest, send, cancel_event):
        for i in range(req.n):
            if cancel_event.is_set():
                return
            await send(MyResponse(progress=i))
        await send(MyResponse(done=True), is_final=True)

    worker = Worker(MyRequest, MyResponse, priority=0, handler=handle)
    asyncio.run(worker.run("ws://localhost:7000/workers/ws"))

The handler is responsible for emitting the final message with
``is_final=True``. If it returns without doing so, the worker emits one
automatically so the server can release the slot.
"""

import asyncio
import ssl
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

import websockets
from pydantic import BaseModel

from .protocol import ServerToWorker, WorkerHello, WorkerToServer

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)

Sender = Callable[[U, bool], Awaitable[None]]
Handler = Callable[[T, Sender, asyncio.Event], Awaitable[None]]


class Worker(Generic[T, U]):
    def __init__(
        self,
        request_type: Type[T],
        response_type: Type[U],
        priority: int,
        handler: Handler,
    ):
        self._request_type = request_type
        self._response_type = response_type
        self._priority = priority
        self._handler = handler
        self._current_task: Optional[asyncio.Task] = None
        self._current_request_id: Optional[str] = None
        self._cancel_event: Optional[asyncio.Event] = None
        self._final_sent: bool = False

    async def run(self, server_url: str) -> None:
        """Connect to the server and process requests forever."""
        ctx = None
        if server_url.startswith("wss://"):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        async with websockets.connect(server_url, max_size=None, ssl=ctx) as ws:
            hello = WorkerHello(priority=self._priority)
            await ws.send(hello.model_dump_json())

            async for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg = ServerToWorker.model_validate_json(raw)

                if msg.type == "cancel":
                    if (
                        self._current_request_id == msg.request_id
                        and self._cancel_event is not None
                    ):
                        self._cancel_event.set()
                    continue

                if msg.type == "request":
                    if self._current_task is not None and not self._current_task.done():
                        # Server should not dispatch to a busy worker; ignore.
                        continue
                    payload = self._request_type.model_validate(msg.payload or {})
                    self._current_request_id = msg.request_id
                    self._cancel_event = asyncio.Event()
                    self._final_sent = False
                    self._current_task = asyncio.create_task(
                        self._run_handler(ws, msg.request_id, payload)
                    )

    async def _run_handler(self, ws, request_id: str, payload: T) -> None:
        async def send(response: U, is_final: bool = False) -> None:
            wrapper = WorkerToServer(
                request_id=request_id,
                payload=response.model_dump(),
                is_final=is_final,
            )
            await ws.send(wrapper.model_dump_json())
            if is_final:
                self._final_sent = True

        try:
            await self._handler(payload, send, self._cancel_event)
        finally:
            if not self._final_sent:
                try:
                    wrapper = WorkerToServer(
                        request_id=request_id, payload={}, is_final=True
                    )
                    await ws.send(wrapper.model_dump_json())
                except Exception:
                    pass
            self._current_request_id = None
            self._cancel_event = None
            self._current_task = None
