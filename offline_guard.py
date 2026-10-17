"""Opt-in pytest guard: block Python socket connections during offline tests.

This is a test-process safeguard, not an OS sandbox or a live-integration check.
Load with: python -m pytest -p offline_guard tests -q
"""
import os
import socket
import threading

_connect = socket.socket.connect
_socketpair = socket.socketpair
_internal = threading.local()

# Avoid loading a nearby private .env before application modules are imported.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ.pop("APPLICATIONINSIGHTS_CONNECTION_STRING", None)
os.environ["OTEL_SDK_DISABLED"] = "true"

def _blocked(*args, **kwargs):
    raise RuntimeError("Network disabled by offline_guard; use a fixture or mock")

def _guarded_connect(sock, address):
    # Windows implements socketpair using a short-lived loopback connection.
    # Only that internal operation is allowed; arbitrary localhost is blocked.
    if getattr(_internal, "socketpair", False) and address[0] in {"127.0.0.1", "::1"}:
        return _connect(sock, address)
    return _blocked()

def _local_socketpair(*args, **kwargs):
    _internal.socketpair = True
    try:
        return _socketpair(*args, **kwargs)
    finally:
        _internal.socketpair = False

def pytest_configure(config):
    # Applied before test collection, including imports of application modules.
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    socket.socketpair = _local_socketpair
