"""Real protocol tests: both servers are spawned as real MCP servers over
stdio and interrogated with the real client library. No API key needed."""
import main


def test_pin_then_verify_clean_server_passes():
    main.pin("server.py")
    result = main.verify("server.py")
    assert result["ok"] is True
    assert result["drifted"] == []


def test_poisoned_server_is_detected_as_drift():
    main.pin("server.py")
    result = main.verify("poisoned_server.py")
    assert result["ok"] is False
    assert "get_price" in result["drifted"]
    # The untouched tool must NOT be flagged: detection is per-tool.
    assert "list_stock" not in result["drifted"]


def test_heuristics_flag_instruction_shaped_language():
    flags = main.heuristic_flags("poisoned_server.py")
    assert "get_price" in flags and len(flags["get_price"]) >= 2
    assert main.heuristic_flags("server.py") == {}
