"""Sanitised strings from the sidecar are JSON strings on the wire."""

from unittest.mock import patch

from acf import Firewall
from acf.models import Decision, SanitiseResult


def test_prompt_sanitised_text_unquotes_wire_json():
    firewall = Firewall(hmac_key=b"x" * 32)
    with patch.object(firewall._transport, "send", return_value={"decision": 1, "sanitised_payload": b'"clean"'}):
        result = firewall.on_prompt("bad")
    assert isinstance(result, SanitiseResult)
    assert result.sanitised_text == "clean"
    assert result.sanitised_payload == b'"clean"'


def test_context_sanitised_text_unquotes_wire_json():
    firewall = Firewall(hmac_key=b"x" * 32)
    with patch.object(firewall._transport, "send", return_value={"decision": 1, "sanitised_payload": b'"[REDACTED]"'}):
        result = firewall.on_context(["bad"])[0]
    assert result.decision == Decision.SANITISE
    assert result.sanitised_text == "[REDACTED]"


def test_structured_sanitised_payload_keeps_json_object():
    firewall = Firewall(hmac_key=b"x" * 32)
    raw = b'{"sanitised":"clean"}'
    with patch.object(firewall._transport, "send", return_value={"decision": 1, "sanitised_payload": raw}):
        result = firewall.on_tool_call("search", {"query": "bad"})
    assert isinstance(result, SanitiseResult)
    assert result.sanitised_text == raw.decode()
