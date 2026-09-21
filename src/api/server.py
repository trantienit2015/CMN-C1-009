"""Standalone HTTP entry point for CMN-C1-009 ClassificationAgent.

Adapter only — no business logic. Provisions secrets via the S-3 provider
pattern and invokes the agent with an InvocationContext. The request is packed
as a JSON string into user_input (parsed by InitializeNode).
"""

from typing import Any
import json
import os
import secrets
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory

from src.graph.graph import ClassificationAgent

app = FastAPI(title="CMN-C1-009 ClassificationAgent")

agent = ClassificationAgent()
agent.compile()
agent.provision_secrets(secrets_factory(namespace="agent1000", agent_name="CMN-C1-009"))


class InvokeRequest(BaseModel):
    input_text: str
    label_list: list[str]
    classification_mode: str = "multi-class"
    few_shot_examples: list[Any] | None = None
    confidence_threshold: float | None = None
    max_input_length: int | None = None
    session_id: str = ""


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Any:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone/STG caller auth: when
    # INVOKE_AUTH_TOKEN is set on the server environment, callers that no upstream
    # middleware vouched for (still ANONYMOUS) must present it as a Bearer token
    # and run at VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary (standalone equivalent of
    # platform AuthMiddleware) — a deployment-level caller credential, not an
    # agent secret, so ctx.secrets does not apply (no InvocationContext exists
    # before auth) — this is a documented entry-point exception.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        payload = json.dumps(
            {
                "input_text": req.input_text,
                "label_list": req.label_list,
                "classification_mode": req.classification_mode,
                "few_shot_examples": req.few_shot_examples,
                "confidence_threshold": req.confidence_threshold,
                "max_input_length": req.max_input_length,
            }
        )
        return agent.invoke(payload, ctx=ctx)


@app.get("/health")
def health() -> Any:
    return {"status": "ok", "agent": "CMN-C1-009"}
