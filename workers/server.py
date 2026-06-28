"""
Generic Server[T, U] that dispatches typed requests to connected workers
over WebSocket.

Usage:
    server = Server(MyRequest, MyResponse)

    # FastAPI worker endpoint
    @app.websocket("/workers/ws")
    async def workers_ws(ws: WebSocket):
        await server.handle_worker(ws)

    # Submitting work from a request handler:
    try:
        request_id = await server.submit(MyRequest(...))
    except NoWorkerAvailable:
        raise HTTPException(status_code=503, detail="no idle worker")

    async for response in server.stream(request_id):
        ...  # forward to client
"""

import asyncio
import uuid
from typing import AsyncIterator, Generic, Optional, Type, TypeVar

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from .protocol import ServerToWorker, WorkerHello, WorkerToServer

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U", bound=BaseModel)


class NoWorkerAvailable(Exception):
    """Raised by Server.submit when no idle worker is available."""


class WorkerDisconnected(Exception):
    """Pushed to a stream when the worker processing it disconnects."""


class _WorkerConnection:
    __slots__ = ("ws", "priority", "current_request_id")

    def __init__(self, ws: WebSocket, priority: int):
        self.ws = ws
        self.priority = priority
        self.current_request_id: Optional[str] = None


class Server(Generic[T, U]):
    def __init__(self, request_type: Type[T], response_type: Type[U]):
        self._request_type = request_type
        self._response_type = response_type
        self._workers: list[_WorkerConnection] = []
        self._streams: dict[str, asyncio.Queue] = {}
        self._assignment: dict[str, _WorkerConnection] = {}
        self._lock = asyncio.Lock()

    # ─── Public API ───────────────────────────────────────────────────────

    async def submit(self, payload: T) -> str:
        """Pick the idle worker with the smallest priority number and send
        the request. Returns a request_id used by ``stream`` / ``cancel``.
        Raises ``NoWorkerAvailable`` if no worker is idle."""
        async with self._lock:
            idle = [w for w in self._workers if w.current_request_id is None]
            if not idle:
                raise NoWorkerAvailable()
            idle.sort(key=lambda w: w.priority)
            worker = idle[0]
            request_id = str(uuid.uuid4())
            worker.current_request_id = request_id
            self._streams[request_id] = asyncio.Queue()
            self._assignment[request_id] = worker

        msg = ServerToWorker(
            request_id=request_id,
            type="request",
            payload=payload.model_dump(),
        )
        try:
            await worker.ws.send_text(msg.model_dump_json())
        except Exception:
            await self._release(request_id, push_disconnect=False)
            raise NoWorkerAvailable()
        return request_id

    async def stream(self, request_id: str) -> AsyncIterator[U]:
        """Yield responses from the worker until ``is_final`` is received,
        or raise ``WorkerDisconnected`` if the worker dies mid-request."""
        queue = self._streams.get(request_id)
        if queue is None:
            return
        try:
            while True:
                item = await queue.get()
                if isinstance(item, WorkerDisconnected):
                    raise item
                if item is None:  # is_final sentinel
                    return
                yield item
        finally:
            self._streams.pop(request_id, None)

    async def cancel(self, request_id: str) -> None:
        """Tell the assigned worker to stop processing ``request_id``."""
        worker = self._assignment.get(request_id)
        if worker is None:
            return
        msg = ServerToWorker(request_id=request_id, type="cancel", payload=None)
        try:
            await worker.ws.send_text(msg.model_dump_json())
        except Exception:
            pass  # disconnect handler will clean up

    # ─── Worker WebSocket handler ────────────────────────────────────────

    async def handle_worker(self, ws: WebSocket) -> None:
        await ws.accept()
        worker: Optional[_WorkerConnection] = None
        try:
            hello_raw = await ws.receive_text()
            hello = WorkerHello.model_validate_json(hello_raw)
            worker = _WorkerConnection(ws=ws, priority=hello.priority)
            async with self._lock:
                self._workers.append(worker)

            while True:
                raw = await ws.receive_text()
                try:
                    msg = WorkerToServer.model_validate_json(raw)
                except ValidationError:
                    continue

                response: Optional[U] = None
                try:
                    response = self._response_type.model_validate(msg.payload)
                except ValidationError:
                    pass  # bad payload — still honor is_final below

                queue = self._streams.get(msg.request_id)
                if queue is not None:
                    if response is not None:
                        await queue.put(response)
                    if msg.is_final:
                        await queue.put(None)

                if msg.is_final:
                    await self._release(msg.request_id, push_disconnect=False)

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if worker is not None:
                async with self._lock:
                    if worker in self._workers:
                        self._workers.remove(worker)
                    orphans = [
                        rid for rid, w in self._assignment.items() if w is worker
                    ]
                for rid in orphans:
                    await self._release(rid, push_disconnect=True)

    # ─── Internal helpers ────────────────────────────────────────────────

    async def _release(self, request_id: str, *, push_disconnect: bool) -> None:
        async with self._lock:
            worker = self._assignment.pop(request_id, None)
            if worker is not None and worker.current_request_id == request_id:
                worker.current_request_id = None
        queue = self._streams.get(request_id)
        if queue is not None and push_disconnect:
            await queue.put(WorkerDisconnected())
            await queue.put(None)

    @property
    def worker_count(self) -> int:
        return len(self._workers)
