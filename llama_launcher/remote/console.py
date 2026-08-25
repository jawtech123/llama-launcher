"""Live log streaming worker.

Follows a server's log by running ``tail -F`` and reading its output. The
connection decides *how* that command reaches the machine (directly, or through
ssh), so the same worker follows a local log and a remote one -- previously this
opened a local file by path and so silently followed nothing for SSH servers.
"""
import logging
import subprocess
from typing import Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class LogFollowWorker(QThread):
    """Streams a server's log file until stopped."""

    line_received = Signal(str)
    error_occurred = Signal(str)
    connected = Signal()
    disconnected = Signal()

    def __init__(self, manager, server_name, initial_lines: int = 50):
        super().__init__()
        self._manager = manager
        self._server_name = server_name
        self._initial_lines = initial_lines
        self._running = False
        self._process: Optional[subprocess.Popen] = None

    def run(self):
        conn = self._manager.get_connection(self._server_name)
        if not conn:
            self.error_occurred.emit(f"Server not found: {self._server_name}")
            self.disconnected.emit()
            return

        try:
            argv = conn.log_follow_argv(self._initial_lines)
        except Exception as e:
            logger.debug("Could not build log follow command", exc_info=True)
            self.error_occurred.emit(str(e))
            self.disconnected.emit()
            return

        self._running = True
        try:
            # tail -F waits for a log that does not exist yet, so starting the
            # console before the server is a valid thing to do.
            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            logger.debug("Could not start log follow process", exc_info=True)
            self.error_occurred.emit(str(e))
            self._running = False
            self.disconnected.emit()
            return

        self.connected.emit()
        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                self.line_received.emit(line.rstrip("\n"))
        except Exception as e:
            if self._running:
                self.error_occurred.emit(str(e))
        finally:
            self._terminate()
            self.disconnected.emit()

    def _terminate(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        except Exception:
            logger.debug("Log follow process cleanup failed", exc_info=True)
        finally:
            if process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    logger.debug("Could not close log stream", exc_info=True)

    def stop(self):
        self._running = False
        # Killing the process unblocks the thread's blocking read on stdout.
        self._terminate()
        self.wait(3000)


#: Previous name, kept so any external caller keeps working.
LocalLogWorker = LogFollowWorker
