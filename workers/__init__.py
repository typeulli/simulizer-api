"""
Generic WebSocket worker-dispatch framework (server + worker halves).

Intentionally a near-verbatim copy of backend-ai/workers so the two services
stay structurally identical. The package __init__ is deliberately empty so that:

  - importing the *worker* half (`workers.worker.Worker`) pulls in only
    `websockets` + `pydantic` (no FastAPI) — keeping the off-box build worker
    (worker.py, run on a Mac) dependency-light, and

  - importing the *server* half (`workers.server.Server`) on the API host pulls
    in FastAPI but NOT `websockets`.

Import the half you need directly:
    from workers.server import Server, NoWorkerAvailable, WorkerDisconnected
    from workers.worker import Worker
"""
