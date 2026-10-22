"""
ZeroMQ IPC bridge — PUSH/PULL socket wrappers.

The PULL socket binds (server role, started by the processing engine).
The PUSH socket connects (client role, started by the capture daemon).
ZeroMQ handles queuing internally if one side starts before the other.
"""

import zmq
import structlog

logger = structlog.get_logger()


class ZmqPusher:
    """Non-blocking PUSH socket for the capture daemon."""

    def __init__(self, ipc_path: str):
        self.ipc_path = ipc_path
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.connect(ipc_path)
        logger.info("zmq_pusher_connected", ipc=ipc_path)

    def send(self, message: str):
        """Send a message without blocking. Silently drops if no receiver."""
        try:
            self.socket.send_string(message, zmq.NOBLOCK)
        except zmq.Again:
            # No receiver connected yet — message is queued internally
            pass

    def close(self):
        """Clean shutdown."""
        self.socket.close()
        self.ctx.term()
        logger.info("zmq_pusher_closed")


class ZmqPuller:
    """Blocking PULL socket for the processing engine."""

    def __init__(self, ipc_path: str):
        self.ipc_path = ipc_path
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(ipc_path)
        logger.info("zmq_puller_bound", ipc=ipc_path)

    def receive(self, timeout_ms: int = 1000) -> str | None:
        """
        Receive a message with timeout.
        Returns the message string, or None if timeout expires.
        """
        if self.socket.poll(timeout_ms):
            return self.socket.recv_string()
        return None

    def close(self):
        """Clean shutdown."""
        self.socket.close()
        self.ctx.term()
        logger.info("zmq_puller_closed")
