from __future__ import annotations

import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointCheck:
    host: str
    port: int
    ok: bool
    error: str = ""


def check_https_endpoints(hosts: list[str], timeout: float = 4.0) -> list[EndpointCheck]:
    checks: list[EndpointCheck] = []
    for host in hosts:
        try:
            with socket.create_connection((host, 443), timeout=timeout):
                checks.append(EndpointCheck(host=host, port=443, ok=True))
        except OSError as exc:
            checks.append(EndpointCheck(host=host, port=443, ok=False, error=str(exc)))
    return checks


def blocked_endpoint_summary(checks: list[EndpointCheck]) -> str:
    failed = [check for check in checks if not check.ok]
    if not failed:
        return ""
    parts = [f"{check.host}: {check.error}" for check in failed]
    return "Outbound HTTPS check failed. " + " | ".join(parts)
