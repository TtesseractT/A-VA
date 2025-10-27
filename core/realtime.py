# core/realtime.py
from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Optional
import zmq

DEFAULT_ADDR = "tcp://127.0.0.1:5557"

@dataclass
class Event:
    t: str
    ts: float
    p: dict

class EventBus:
    """
    Tiny PUB/SUB bus over ZeroMQ.
    Chat app = publisher. Dashboard = subscriber.
    """
    def __init__(self, role: str, addr: str = DEFAULT_ADDR):
        self.addr = addr
        self.ctx = zmq.Context.instance()
        if role == "pub":
            self.sock = self.ctx.socket(zmq.PUB)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.bind(addr)
            self.role = "pub"
        elif role == "sub":
            self.sock = self.ctx.socket(zmq.SUB)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(addr)
            self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
            self.role = "sub"
        else:
            raise ValueError("role must be 'pub' or 'sub'")

    def publish(self, typ: str, payload: dict):
        if self.role != "pub":
            return
        msg = json.dumps({"t": typ, "ts": time.time(), "p": payload}, ensure_ascii=False)
        self.sock.send_string(msg)

    def recv(self, timeout_ms: int = 50) -> Optional[Event]:
        if self.role != "sub":
            return None
        try:
            if self.sock.poll(timeout_ms):
                raw = self.sock.recv_string()
                d = json.loads(raw)
                return Event(d.get("t", ""), float(d.get("ts", 0)), d.get("p", {}) or {})
        except Exception:
            return None
        return None
