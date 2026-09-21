"""S-3 output gate — credential / PII detection over the classification result.

Template-owned (developer guide §5: content masking/scanning is plain service
logic, not a framework gate node). The framework does not ship
``framework.security``. Pure functions, no state.
"""

from __future__ import annotations

import re

_OUTPUT_PII_PATTERNS = [
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("phone_jp", re.compile(r"(?:0[0-9]{1,4}-[0-9]{1,4}-[0-9]{4}|\+81[0-9]{9,10})")),
]
_OUTPUT_CREDENTIAL_PATTERNS = [
    ("api_key_openai", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    ("jwt", re.compile(r"eyJ[a-zA-Z0-9._\-]{10,}")),
    ("aws_access_key", re.compile(r"AKIA[A-Z0-9]{16}")),
]


def scan_output(text: str) -> tuple[str, str] | None:
    """Return (error_code, error_message) if the output leaks PII/credentials, else None."""
    if not text:
        return None
    for pii_type, pat in _OUTPUT_PII_PATTERNS:
        if pat.search(text):
            return ("S3_OUTPUT_PII_DETECTED", f"S-3 gate: PII in output (type: {pii_type}).")
    for cred_type, pat in _OUTPUT_CREDENTIAL_PATTERNS:
        if pat.search(text):
            return ("S3_OUTPUT_CREDENTIAL_DETECTED", f"S-3 gate: credential in output (type: {cred_type}).")
    return None
