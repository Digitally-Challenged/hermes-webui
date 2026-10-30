"""#8613: follow-ups to the profile-scoped memory fix (#8517 / PR #8585).

1. A session-end commit the Agent refuses at hook time is detected through the
   Agent's refusal WARNING on ``agent.memory_caller_scope``. With the root logger
   (or ``agent``/that logger) above WARNING, or that logger disabled by a
   ``dictConfig``, the record was never created, so the refused commit was counted
   as done and never retried. The capture must see it whatever the deployer's
   levels are, without changing what the deployer's handlers receive, and must
   restore the logger's state under concurrent and nested captures.
2. Evicting an agent whose session owner cannot be resolved refuses its
   ``shutdown_memory_provider`` (correct: session end commits), but that also
   skipped the provider teardown, so the provider's ``mem-sync`` writer thread
   lingered until the process exited. The provider is now released through the
   Agent's ``release_soft`` in its own bound profile, with no session end, and only
   in that case.
3. docs/troubleshooting.md documents the implemented warnings.

Every test below except the explicit guards was RED on origin/master 72ac887e3.
"""
# ruff: noqa: F811  (pytest fixtures imported from the #8517 modules)
from __future__ import annotations

import ast
import logging
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.memory_profile_scope as scope
import api.session_lifecycle as session_lifecycle
import api.streaming as streaming
from tests.test_issue8517_profile_scoped_memory import (  # noqa: F401 (fixtures)
    _key,
    hc,
    homes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_LOGGER = "agent.memory_caller_scope"


def _caller_home() -> str:
    return str(sys.modules["hermes_constants"].get_hermes_home())


def _agent_refuses(hook: str = "on_session_end") -> None:
    """The Agent's hook-time refusal (agent/memory_caller_scope._refuse): the real one
    when installed, else the same WARNING on the same logger."""
    try:
        from agent.memory_caller_scope import _refuse  # noqa: PLC0415
    except ImportError:
        logging.getLogger(SCOPE_LOGGER).warning(
            "Memory %s refused (%s): %s; caller profile home %r, provider bound to %r, multiplexed=%s",
            f"hook {hook}", "memory_caller_profile_unresolved", "test refusal", None, None, False,
        )
        return
    _refuse(f"hook {hook}", "test refusal", None, None, False)


def _logger_state(lg: logging.Logger) -> tuple:
    return (lg.level, lg.disabled, lg.propagate, list(lg.filters), list(lg.handlers))


@pytest.fixture
def lifecycle(monkeypatch):
    session_lifecycle._reset_for_tests()
    monkeypatch.setattr(session_lifecycle, "_draining", False)
    yield session_lifecycle
    session_lifecycle._reset_for_tests()


@pytest.fixture
def log_config():
    """Restores every logger this module raises or disables."""
    names = ["", "agent", SCOPE_LOGGER]
    saved = {n: (logging.getLogger(n).level, logging.getLogger(n).disabled) for n in names}
    yield
    for n, (level, disabled) in saved.items():
        lg = logging.getLogger(n)
        lg.setLevel(level)
        lg.disabled = disabled
    assert getattr(scope, "_capture_window", None) is None, "a refusal capture window was left open"


class _Sink(logging.Handler):
    """A deployer's handler: what the deployer's logging configuration emits."""

    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _raise_levels(how: str) -> None:
    if how == "root_error":
        logging.getLogger().setLevel(logging.ERROR)
    elif how == "agent_critical":
        logging.getLogger("agent").setLevel(logging.CRITICAL)
    elif how == "scope_logger_error":
        logging.getLogger(SCOPE_LOGGER).setLevel(logging.ERROR)
    elif how == "scope_logger_disabled":  # dictConfig(disable_existing_loggers=True)
        logging.getLogger(SCOPE_LOGGER).disabled = True
    else:  # pragma: no cover
        raise AssertionError(how)


RAISED = ["root_error", "agent_critical", "scope_logger_error", "scope_logger_disabled"]


class RefusingManager:
    """MemoryManager lifecycle stand-in for an Agent whose binding WebUI cannot read
    (no ``_bound_scope``): WebUI runs the hook and the Agent refuses it with its
    WARNING, the only signal."""

    def __init__(self, log, refuse):
        self._providers = ["sbxmem"]
        self.log = log
        self.refuse = refuse

    def on_session_end(self, messages):
        if self.refuse["on"]:
            _agent_refuses("on_session_end")
            return
        self.log.append(("on_session_end", _caller_home()))


class CommitAgent:
    def __init__(self, manager):
        self._memory_manager = manager
        self._session_messages = []
        self._session_db = None

    def commit_memory_session(self, messages=None):
        if self._memory_manager:
            self._memory_manager.on_session_end(messages or [])


def _bound_agent(home, fn_log=None):
    return SimpleNamespace(_memory_manager=SimpleNamespace(
        _bound_scope={"hermes_home": str(home), "home_key": _key(home)}))


# ── 1. refusal accounting does not depend on the deployer's logger levels ──────


@pytest.mark.parametrize("how", RAISED)
def test_refused_commit_stays_pending_whatever_the_logger_levels(hc, homes, lifecycle, log_config, how):
    """Re-gate probe a4: the root logger above WARNING; the refusal was missed and the
    commit counted as done. Same for ``agent``/the scope logger raised or disabled."""
    log: list = []
    refuse = {"on": True}
    sid = f"i8613-raised-{how}"
    agent = CommitAgent(RefusingManager(log, refuse))
    lifecycle.mark_turn_completed(sid, agent=agent, profile_home=str(homes["alpha"]))
    scope_logger = logging.getLogger(SCOPE_LOGGER)
    sink = _Sink()
    root = logging.getLogger()
    root.addHandler(sink)
    try:
        _raise_levels(how)
        before = _logger_state(scope_logger)
        assert lifecycle.commit_session_memory(sid, wait=True) is False
        assert _logger_state(scope_logger) == before, "the scope logger's state must be restored"
    finally:
        root.removeHandler(sink)
    assert lifecycle.has_uncommitted_work(sid), "a refused commit must stay pending"
    assert log == []
    # The deployer's handlers saw exactly what their configuration emits: nothing.
    assert not [r for r in sink.records if r.name == SCOPE_LOGGER], sink.records

    refuse["on"] = False
    assert lifecycle.commit_session_memory(sid, wait=True) is True
    assert not lifecycle.has_uncommitted_work(sid)
    assert log == [("on_session_end", str(homes["alpha"]))]


def test_refusal_line_still_reaches_the_deployer_at_default_levels(hc, homes, log_config):
    """GREEN guard: with WARNING admitted the deployer's handlers keep receiving it and
    the logger is not touched."""
    logging.getLogger().setLevel(logging.WARNING)
    scope_logger = logging.getLogger(SCOPE_LOGGER)
    before = _logger_state(scope_logger)
    sink = _Sink()
    logging.getLogger().addHandler(sink)
    seen: list = []
    try:
        ok = scope.call_as_session_owner(
            _bound_agent(homes["alpha"]),
            lambda: (seen.append(_logger_state(scope_logger)[:3]), _agent_refuses()),
            session_id="i8613-default", owner_home=str(homes["alpha"]),
        )
    finally:
        logging.getLogger().removeHandler(sink)
    assert ok is False
    assert seen == [before[:3]], "a logger that already admits WARNING must not be changed"
    assert [r for r in sink.records if r.name == SCOPE_LOGGER and "refused" in r.getMessage()]
    assert _logger_state(scope_logger) == before


def test_real_agent_refusal_under_a_raised_root_level_stays_pending(hc, homes, lifecycle, log_config, monkeypatch):
    """The installed Agent's own MemoryManager refusing ``on_session_end`` (bound to
    beta, called as alpha) on an Agent whose binding WebUI cannot read."""
    mm_mod = pytest.importorskip("agent.memory_manager")
    from agent.memory_provider import MemoryProvider  # noqa: PLC0415

    events: list = []

    class Recorder(MemoryProvider):
        name = "rec8613"

        def is_available(self):
            return True

        def get_tool_schemas(self):
            return []

        def initialize(self, session_id, **kwargs):
            events.append(("initialize", kwargs.get("hermes_home")))

        def on_session_end(self, messages):
            events.append(("on_session_end", _caller_home()))

    manager = mm_mod.MemoryManager()
    manager.add_provider(Recorder())
    manager.initialize_all(session_id="s", hermes_home=str(homes["beta"]))
    monkeypatch.setattr(scope, "_bound_memory_home_key", lambda agent: ("unverifiable", None))
    agent = CommitAgent(manager)
    sid = "i8613-real-refusal"
    lifecycle.mark_turn_completed(sid, agent=agent, profile_home=str(homes["alpha"]))
    logging.getLogger().setLevel(logging.ERROR)
    assert lifecycle.commit_session_memory(sid, wait=True) is False
    assert lifecycle.has_uncommitted_work(sid)
    assert ("on_session_end", str(homes["alpha"])) not in events and not [
        e for e in events if e[0] == "on_session_end"]


def test_concurrent_sessions_capture_only_their_own_refusal_and_restore_the_logger(hc, homes, log_config):
    """Two sessions' commits overlap on two threads with the root above WARNING. B is
    refused while A is open; A must not take B's refusal, B must see it, and the
    logger is restored only when the LAST capture closes (A closes first here)."""
    logging.getLogger().setLevel(logging.ERROR)
    scope_logger = logging.getLogger(SCOPE_LOGGER)
    before = _logger_state(scope_logger)
    a_open, b_refused, a_closed = threading.Event(), threading.Event(), threading.Event()
    results: dict = {}
    during: dict = {}

    def a_fn():
        a_open.set()
        assert b_refused.wait(5)

    def b_fn():
        assert a_open.wait(5)
        _agent_refuses()
        b_refused.set()
        assert a_closed.wait(5)
        during["b_after_a_closed"] = scope_logger.isEnabledFor(logging.WARNING)

    def run(name, agent, fn, home):
        try:
            results[name] = scope.call_as_session_owner(agent, fn, session_id=f"i8613-{name}", owner_home=str(home))
        except BaseException as exc:  # surfaced below
            results[name] = exc
        finally:
            if name == "a":
                a_closed.set()

    threads = [
        threading.Thread(target=run, args=("a", _bound_agent(homes["alpha"]), a_fn, homes["alpha"])),
        threading.Thread(target=run, args=("b", _bound_agent(homes["beta"]), b_fn, homes["beta"])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == {"a": True, "b": False}
    assert during == {"b_after_a_closed": True}, "A's close must not end B's capture window"
    assert _logger_state(scope_logger) == before
    assert not scope_logger.isEnabledFor(logging.WARNING)


def test_nested_captures_restore_the_logger(hc, homes, log_config):
    logging.getLogger("agent").setLevel(logging.CRITICAL)
    scope_logger = logging.getLogger(SCOPE_LOGGER)
    before = _logger_state(scope_logger)
    inner: dict = {}

    def outer_fn():
        inner["ok"] = scope.call_as_session_owner(
            _bound_agent(homes["alpha"]), _agent_refuses, session_id="i8613-inner", owner_home=str(homes["alpha"]))
        inner["still_open"] = scope_logger.isEnabledFor(logging.WARNING)

    outer = scope.call_as_session_owner(
        _bound_agent(homes["alpha"]), outer_fn, session_id="i8613-outer", owner_home=str(homes["alpha"]))
    assert inner == {"ok": False, "still_open": True}
    # The inner refusal ran inside the outer call too: the outer is not counted either.
    assert outer is False
    assert _logger_state(scope_logger) == before


def test_capture_restores_the_logger_when_the_hook_raises(hc, homes, log_config):
    logging.getLogger().setLevel(logging.ERROR)
    scope_logger = logging.getLogger(SCOPE_LOGGER)
    before = _logger_state(scope_logger)

    def boom():
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError):
        scope.call_as_session_owner(_bound_agent(homes["alpha"]), boom, session_id="i8613-raise",
                                    owner_home=str(homes["alpha"]))
    assert _logger_state(scope_logger) == before


def test_reconfiguration_during_the_window_is_kept(hc, homes, log_config):
    """Restore is compare-and-restore: a level the deployer sets meanwhile wins."""
    logging.getLogger().setLevel(logging.ERROR)
    scope_logger = logging.getLogger(SCOPE_LOGGER)

    def reconfigure():
        scope_logger.setLevel(logging.INFO)

    scope.call_as_session_owner(_bound_agent(homes["alpha"]), reconfigure, session_id="i8613-reconf",
                                owner_home=str(homes["alpha"]))
    assert scope_logger.level == logging.INFO


# ── 2. unresolvable-owner eviction releases the provider, never as a session end ──


def _manager(home, log, *, release=True, claim=True, bound=True):
    """MemoryManager stand-in recording every lifecycle call and the caller's home."""
    m = SimpleNamespace(_providers=["sbxmem"])
    if bound:
        m._bound_scope = {"hermes_home": str(home), "home_key": _key(home)}
    m.on_session_end = lambda messages: log.append(("on_session_end", _caller_home()))
    m.shutdown_all = lambda: log.append(("shutdown_all", _caller_home()))
    if claim:
        m.claim_teardown = lambda *, session_end: log.append(("claim_teardown", session_end)) or True
    if release:
        m.release_soft = lambda: log.append(("release_soft", _caller_home()))
    return m


class EvictedAgent:
    def __init__(self, manager, log):
        self._memory_manager = manager
        self._session_messages = [{"role": "user", "content": "x"}]
        self._session_db = None
        self.log = log

    def commit_memory_session(self, messages=None):
        self.log.append(("commit_memory_session", _caller_home()))

    def shutdown_memory_provider(self, messages=None):
        self.log.append(("shutdown_memory_provider", _caller_home()))


def _evict_from(hc, home, sid, agent):
    """Evict ``agent`` from another profile's turn (``home``'s override installed)."""
    token = hc.set_hermes_home_override(str(home))
    try:
        closed = streaming._close_evicted_agent_at_session_boundary(sid, agent)
        assert _caller_home() == str(home), "the evicting turn's scope must be restored"
    finally:
        hc.reset_hermes_home_override(token)
    return closed


def test_unresolvable_owner_eviction_releases_the_provider_in_its_own_profile(hc, homes, lifecycle):
    """Re-gate probe b: no recorded owner, no session record. The shutdown is refused
    (correct) but the provider was never released."""
    log: list = []
    sid = "i8613-unowned"
    agent = EvictedAgent(_manager(homes["alpha"], log), log)
    assert scope.session_owner_profile_home(sid) == ""
    assert _evict_from(hc, homes["beta"], sid, agent) is True
    assert log == [("release_soft", str(homes["alpha"]))], (
        "only the provider release may run, in the profile it is bound to, never a "
        f"session end / commit / shutdown as the evicting profile: {log}"
    )


@pytest.mark.parametrize("missing", ["release", "claim", "bound"])
def test_unresolvable_owner_without_a_safe_release_keeps_skipping_and_logs_once(
    hc, homes, lifecycle, missing, caplog, monkeypatch
):
    monkeypatch.setattr(scope, "_logged_once", set())
    caplog.set_level(logging.WARNING, logger="api.memory_profile_scope")
    log: list = []
    for n in range(2):
        agent = EvictedAgent(_manager(homes["alpha"], log, **{missing: False}), log)
        assert _evict_from(hc, homes["beta"], f"i8613-nosafe-{missing}-{n}", agent) is True
    assert log == [], log
    lines = [r.getMessage() for r in caplog.records if "#8613" in r.getMessage()]
    assert len(lines) == 1, lines


def test_known_owner_eviction_is_unchanged_and_never_soft_released(hc, homes, lifecycle):
    """GREEN guard: the owner is known, the normal shutdown runs as that owner."""
    log: list = []
    sid = "i8613-owned"
    agent = EvictedAgent(_manager(homes["alpha"], log), log)
    lifecycle.record_owner_profile_home(sid, str(homes["alpha"]))
    assert _evict_from(hc, homes["beta"], sid, agent) is True
    assert log == [("shutdown_memory_provider", str(homes["alpha"]))], log


def test_owner_known_but_provider_bound_elsewhere_is_not_released(hc, homes, lifecycle):
    """GREEN guard: a refusal for any reason but an unknown owner leaves the provider
    alone (it is bound to another profile than the session's)."""
    log: list = []
    sid = "i8613-misbound"
    agent = EvictedAgent(_manager(homes["beta"], log), log)
    lifecycle.record_owner_profile_home(sid, str(homes["alpha"]))
    assert _evict_from(hc, homes["beta"], sid, agent) is True
    assert log == [], log


def test_release_helper_refuses_when_the_owner_resolves(hc, homes):
    log: list = []
    agent = EvictedAgent(_manager(homes["alpha"], log), log)
    assert scope.release_unowned_memory_provider(agent, session_id="x", owner_home=str(homes["alpha"])) is False
    assert scope.release_unowned_memory_provider(SimpleNamespace(), session_id="x") is False
    assert log == []


def test_real_memory_manager_unowned_eviction_leaves_no_mem_sync_thread(hc, homes, lifecycle):
    """The installed Agent: a provider with a live ``mem-sync`` writer, evicted with an
    unresolvable owner. The writer stops, queued writes land in the provider's own
    profile, no session end runs, and a later teardown does not shut it down twice."""
    mm_mod = pytest.importorskip("agent.memory_manager")
    from agent.memory_provider import MemoryProvider  # noqa: PLC0415

    if not (callable(getattr(mm_mod.MemoryManager, "release_soft", None))
            and callable(getattr(mm_mod.MemoryManager, "claim_teardown", None))):
        pytest.skip("installed Agent has no claimed soft release")
    events: list = []

    class Recorder(MemoryProvider):
        name = "rec8613"

        def is_available(self):
            return True

        def get_tool_schemas(self):
            return []

        def initialize(self, session_id, **kwargs):
            events.append(("initialize", kwargs.get("hermes_home")))

        def sync_turn(self, user_content, assistant_content, **kwargs):
            events.append(("sync_turn", _caller_home()))

        def on_session_end(self, messages):
            events.append(("on_session_end", _caller_home()))

        def shutdown(self):
            events.append(("shutdown", _caller_home()))

    manager = mm_mod.MemoryManager()
    manager.add_provider(Recorder())
    manager.initialize_all(session_id="s", hermes_home=str(homes["alpha"]))
    token = hc.set_hermes_home_override(str(homes["alpha"]))  # alpha's own turn
    try:
        manager.sync_all("hello", "world", session_id="s")
        assert manager.flush_pending(timeout=5)
    finally:
        hc.reset_hermes_home_override(token)
    executor = manager._sync_executor
    workers = [t for t in getattr(executor, "_threads", ()) if t.is_alive()]
    assert workers and all(t.name.startswith("mem-sync") for t in workers)

    log: list = []
    agent = EvictedAgent(manager, log)
    assert _evict_from(hc, homes["beta"], "i8613-real-unowned", agent) is True
    for t in workers:
        t.join(5)
    assert not [t for t in workers if t.is_alive()], "the evicted agent's mem-sync worker lingers"
    assert log == [], "no session end / commit / shutdown_memory_provider as an unknown owner"
    assert [e for e in events if e[0] != "initialize"] == [
        ("sync_turn", str(homes["alpha"])),
        ("shutdown", str(homes["alpha"])),
    ], events
    # Provider idempotency: the claim makes any later teardown a no-op.
    manager.release_soft()
    assert manager.claim_teardown(session_end=True) is False
    assert [e for e in events if e[0] == "shutdown"] == [("shutdown", str(homes["alpha"]))]


def test_streaming_releases_only_after_a_refused_shutdown():
    src = (REPO_ROOT / "api" / "streaming.py").read_text(encoding="utf-8")
    m = re.search(r"def _close_evicted_agent_at_session_boundary\(.*?\n(?=def )", src, re.S)
    assert m, "eviction helper not found"
    body = m.group(0)
    assert "release_unowned_memory_provider(" in body
    assert body.index("call_as_session_owner(") < body.index("release_unowned_memory_provider(")


# ── 3. docs: the implemented warnings are documented, verbatim ─────────────────


def _string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_troubleshooting_documents_the_memory_scope_warnings():
    doc = (REPO_ROOT / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    m = re.search(r"\n## Memory is off.*?(?=\n## )", doc, re.S)
    assert m, "docs/troubleshooting.md has no memory-scope entry"
    entry = m.group(0)
    for needle in ("v0.18.0", "claim_instance", "first profile", "pending", "#8517", "#8613"):
        assert needle in entry, needle
    # Every log fragment the entry quotes is one the code actually logs.
    constants = _string_constants(REPO_ROOT / "api" / "memory_profile_scope.py")
    prose = re.sub(r"```.*?```", "", entry, flags=re.S)
    quoted = [span for span in re.findall(r"`([^`]+)`", prose)
              if re.search(r"refused|older than|shared provider instance|session end|pending|restarts", span)]
    assert len(quoted) >= 6, quoted
    for fragment in quoted:
        assert any(fragment in c for c in constants), f"documented log text not in the code: {fragment!r}"
