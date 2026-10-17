"""Ensure the opt-in offline command cannot open a provider connection."""
import socket
import sys
import pytest

@pytest.mark.skipif("offline_guard" not in sys.modules, reason="requires -p offline_guard")
def test_network_is_blocked():
    with pytest.raises(RuntimeError, match="Network disabled"):
        socket.create_connection(("example.invalid", 443))

@pytest.mark.skipif("offline_guard" not in sys.modules, reason="requires -p offline_guard")
def test_local_socketpair_works_but_direct_loopback_is_blocked():
    left, right = socket.socketpair()
    try:
        left.send(b"fixture")
        assert right.recv(7) == b"fixture"
    finally:
        left.close()
        right.close()
    with socket.socket() as client:
        with pytest.raises(RuntimeError, match="Network disabled"):
            client.connect(("127.0.0.1", 443))
