"""Page types are declared per wiki in SCHEMA.md, and multiple wikis may be
tracked at once.

Covers the two extensions to LLM Wiki status:

1. A wiki declares its own page sections under a "## Page Types" heading in
   SCHEMA.md. The four canonical sections remain the fallback, so a wiki with
   no SCHEMA.md behaves exactly as it did before.
2. Additional wikis are opted into via the *plural* knobs (``WIKI_PATHS`` env
   or ``skills.config.wiki.paths``). ``WIKI_PATH``/``wiki.path`` still selects
   the single primary wiki with unchanged precedence.

The privacy contract from test_issue1257_llm_wiki_status (the status payload
never carries a wiki's filesystem path) is re-asserted here for the multi-wiki
payload.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _write(path: Path, text: str = "# Synthetic\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _health_like_wiki(root: Path) -> Path:
    """A wiki declaring non-canonical page types, mirroring the health wiki."""
    _write(
        root / "SCHEMA.md",
        "# Wiki Schema\n\n"
        "## Page Types\n\n"
        "- `entities/` → `entity` — orgs and people\n"
        "- `compounds/` → `compound` — substances\n"
        "- `biomarkers/` → `biomarker` — bloodwork\n"
        "- `concepts/` → `concept` — mechanisms\n"
        "- `queries/` → `query` — filed answers\n\n"
        "## Tag Taxonomy\n\n- `safety` — risk\n",
    )
    _write(root / "index.md", "# Index\n")
    _write(root / "log.md", "# Log\n")
    _write(root / "compounds" / "psilocybin.md", "---\ntitle: Psilocybin\n---\nbody\n")
    _write(root / "biomarkers" / "apob.md", "---\ntitle: ApoB\n---\nbody\n")
    _write(root / "concepts" / "integration.md", "---\ntitle: Integration\n---\nbody\n")
    return root


# ── Page-type parsing ─────────────────────────────────────────────────────────


def test_page_dirs_parsed_from_schema(tmp_path):
    from api import routes

    wiki = _health_like_wiki(tmp_path / "wiki")
    dirs, types = routes._llm_wiki_page_dirs(wiki)

    assert list(dirs) == ["entities", "compounds", "biomarkers", "concepts", "queries"]
    assert types["compounds"] == "compound"
    assert types["biomarkers"] == "biomarker"


def test_page_dirs_accepts_ascii_arrow_and_colon(tmp_path):
    from api import routes

    wiki = tmp_path / "wiki"
    _write(
        wiki / "SCHEMA.md",
        "# Schema\n\n## Page Types\n\n"
        "- `alpha/` -> `a` — ascii arrow\n"
        "- `beta/`: `b` — colon form\n",
    )
    dirs, types = routes._llm_wiki_page_dirs(wiki)

    assert list(dirs) == ["alpha", "beta"]
    assert types == {"alpha": "a", "beta": "b"}


def test_page_dirs_fall_back_without_schema(tmp_path):
    from api import routes

    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)

    dirs, types = routes._llm_wiki_page_dirs(wiki)

    assert list(dirs) == ["entities", "concepts", "comparisons", "queries"]
    assert types == {}


def test_page_dirs_fall_back_when_schema_has_no_page_types(tmp_path):
    from api import routes

    wiki = tmp_path / "wiki"
    _write(wiki / "SCHEMA.md", "# Schema\n\n## Domain\n\nSomething.\n")

    dirs, _ = routes._llm_wiki_page_dirs(wiki)

    assert list(dirs) == ["entities", "concepts", "comparisons", "queries"]


@pytest.mark.parametrize(
    "row",
    [
        "- `../etc/` → `evil` — traversal",
        "- `raw/` → `raw` — must never be a page section",
        "- `_archive/` → `archive` — must never be a page section",
        "- `.hidden/` → `hidden` — dotfiles excluded",
        "- `a/b/` → `nested` — single segment only",
    ],
)
def test_page_types_reject_unsafe_directory_names(tmp_path, row):
    from api import routes

    wiki = tmp_path / "wiki"
    _write(wiki / "SCHEMA.md", f"# Schema\n\n## Page Types\n\n{row}\n")

    dirs, _ = routes._llm_wiki_page_dirs(wiki)

    # Nothing usable declared → canonical fallback, never the unsafe name.
    assert list(dirs) == ["entities", "concepts", "comparisons", "queries"]


def test_page_types_stops_at_next_heading(tmp_path):
    """A 'dir/ → type' line outside the Page Types section must be ignored."""
    from api import routes

    wiki = tmp_path / "wiki"
    _write(
        wiki / "SCHEMA.md",
        "# Schema\n\n## Page Types\n\n- `compounds/` → `compound` — in section\n\n"
        "## Something Else\n\n- `sneaky/` → `sneaky` — outside the section\n",
    )

    dirs, _ = routes._llm_wiki_page_dirs(wiki)

    assert list(dirs) == ["compounds"]


# ── The walk honours declared page types ──────────────────────────────────────


def test_page_files_finds_declared_non_canonical_sections(tmp_path):
    from api import routes

    wiki = _health_like_wiki(tmp_path / "wiki")
    routes._llm_wiki_clear_page_files_cache()

    found = {p.name for p in routes._llm_wiki_page_files(wiki)}

    # compounds/ and biomarkers/ are not canonical, but are declared.
    assert found == {"psilocybin.md", "apob.md", "integration.md"}


def test_raw_is_never_counted_as_a_page(tmp_path):
    from api import routes

    wiki = _health_like_wiki(tmp_path / "wiki")
    _write(wiki / "raw" / "articles" / "source.md", "raw body\n")
    routes._llm_wiki_clear_page_files_cache()

    found = {p.name for p in routes._llm_wiki_page_files(wiki)}

    assert "source.md" not in found


def test_undeclared_directory_is_not_walked(tmp_path):
    """A section the schema doesn't declare is invisible, as documented."""
    from api import routes

    wiki = _health_like_wiki(tmp_path / "wiki")
    _write(wiki / "products" / "mystery.md", "---\ntitle: M\n---\nbody\n")
    routes._llm_wiki_clear_page_files_cache()

    found = {p.name for p in routes._llm_wiki_page_files(wiki)}

    assert "mystery.md" not in found


# ── Multi-wiki resolution ─────────────────────────────────────────────────────


def test_single_wiki_resolution_is_unchanged(tmp_path, monkeypatch):
    """With no plural knob set, resolution is exactly the legacy single wiki."""
    from api import routes

    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    monkeypatch.setenv("WIKI_PATH", str(wiki))
    monkeypatch.delenv("WIKI_PATHS", raising=False)
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])

    resolved = routes._llm_wiki_resolve_paths()

    assert len(resolved) == 1
    assert resolved[0] == routes._llm_wiki_resolve_path()
    assert resolved[0][0] == wiki


def test_plural_env_adds_extra_wikis(tmp_path, monkeypatch):
    from api import routes

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (primary / "concepts").mkdir(parents=True)
    (extra / "concepts").mkdir(parents=True)
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.setenv("WIKI_PATHS", f"{extra}, {primary}")  # primary duped on purpose
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])

    resolved = routes._llm_wiki_resolve_paths()

    assert [str(p) for p, _, _ in resolved] == [str(primary), str(extra)]
    assert resolved[0][1] == "WIKI_PATH"
    assert resolved[1][1] == "WIKI_PATHS"


def test_config_path_list_adds_extra_wikis(tmp_path, monkeypatch):
    from api import routes

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (primary / "concepts").mkdir(parents=True)
    (extra / "concepts").mkdir(parents=True)
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.delenv("WIKI_PATHS", raising=False)
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [str(extra)])

    resolved = routes._llm_wiki_resolve_paths()

    assert [str(p) for p, _, _ in resolved] == [str(primary), str(extra)]
    assert resolved[1][1] == "skills.config.wiki.paths"


# ── Aggregated status payload ─────────────────────────────────────────────────


def test_aggregate_reports_counts_across_wikis(tmp_path, monkeypatch):
    from api import routes

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (primary / "concepts").mkdir(parents=True)
    _health_like_wiki(extra)
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.setenv("WIKI_PATHS", str(extra))
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])
    routes._llm_wiki_clear_page_files_cache()

    status = routes._build_llm_wiki_status()

    assert status["wiki_count"] == 2
    assert [w["label"] for w in status["wikis"]] == ["primary", "extra"]
    # Top-level keys describe the primary wiki only — unchanged shape.
    assert status["page_count"] == 0
    assert status["page_count_total"] == 3
    assert status["entry_count_total"] == 3
    assert status["wikis"][1]["page_count"] == 3


def test_single_wiki_payload_is_a_superset_of_the_legacy_shape(tmp_path, monkeypatch):
    from api import routes

    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    monkeypatch.setenv("WIKI_PATH", str(wiki))
    monkeypatch.delenv("WIKI_PATHS", raising=False)
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])

    status = routes._build_llm_wiki_status()

    for key in (
        "available", "enabled", "status", "entry_count", "page_count",
        "raw_source_count", "last_updated", "last_writer", "path_configured",
        "path_source", "toggle_available", "toggle_reason", "docs_url",
    ):
        assert key in status, f"legacy key {key!r} disappeared from the payload"
    assert status["wiki_count"] == 1
    assert status["wikis"][0]["status"] == status["status"]


def test_multi_wiki_payload_never_carries_a_filesystem_path(tmp_path, monkeypatch):
    """Extends test_issue1257's privacy contract to the multi-wiki payload."""
    from api import routes

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (primary / "concepts").mkdir(parents=True)
    (extra / "concepts").mkdir(parents=True)
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.setenv("WIKI_PATHS", str(extra))
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])

    status = routes._build_llm_wiki_status()
    serialized = repr(status)

    assert str(primary) not in serialized
    assert str(extra) not in serialized
    assert str(tmp_path) not in serialized


def test_one_broken_wiki_does_not_blank_the_others(tmp_path, monkeypatch):
    """Per-wiki error isolation: a missing extra must not fail the primary."""
    from api import routes

    primary = tmp_path / "primary"
    (primary / "concepts").mkdir(parents=True)
    (primary / "concepts" / "one.md").write_text("---\ntitle: One\n---\nbody\n", encoding="utf-8")
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.setenv("WIKI_PATHS", str(tmp_path / "nope"))
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])
    routes._llm_wiki_clear_page_files_cache()

    status = routes._build_llm_wiki_status()

    assert status["status"] == "ready"
    assert status["page_count"] == 1
    assert status["wikis"][1]["status"] == "missing"
    assert status["wikis"][1]["available"] is False


def test_status_reports_declared_page_dirs(tmp_path, monkeypatch):
    from api import routes

    wiki = _health_like_wiki(tmp_path / "wiki")
    monkeypatch.setenv("WIKI_PATH", str(wiki))
    monkeypatch.delenv("WIKI_PATHS", raising=False)
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])
    routes._llm_wiki_clear_page_files_cache()

    status = routes._build_llm_wiki_status()

    assert "compounds" in status["page_dirs"]
    assert status["page_count"] == 3


# ── Multi-wiki BROWSE / PAGE (index-based selection) ──────────────────────────


class _FakeHandler:
    def __init__(self):
        self.status = None
        self.body = bytearray()
        self.wfile = self

    def send_response(self, code):
        self.status = code

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass

    def write(self, data):
        self.body.extend(data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8"))

    def get_json(self):
        import json
        return json.loads(self.body.decode("utf-8"))


def _two_wikis(tmp_path, monkeypatch):
    """A primary wiki and an extra wiki, both with one page each."""
    from api import routes
    from urllib.parse import urlparse  # noqa: F401  (kept for callers)

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    _health_like_wiki(primary)      # compounds/psilocybin.md, biomarkers/apob.md, concepts/integration.md
    (extra / "concepts").mkdir(parents=True)
    (extra / "concepts" / "only-in-extra.md").write_text(
        "---\ntitle: Extra\n---\nextra body\n", encoding="utf-8"
    )
    monkeypatch.setenv("WIKI_PATH", str(primary))
    monkeypatch.setenv("WIKI_PATHS", str(extra))
    monkeypatch.setattr(routes, "_llm_wiki_config_path_list", lambda: [])
    routes._llm_wiki_clear_page_files_cache()
    return primary, extra


def test_browse_without_index_uses_primary(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    primary, _extra = _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse("http://x/api/wiki/browse"))

    assert h.status == 200
    assert "compounds/psilocybin.md" in [p["path"] for p in h.get_json()["pages"]]


def test_browse_wiki_zero_explicitly_matches_omitted(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)
    h0, h1 = _FakeHandler(), _FakeHandler()
    routes.handle_get(h0, urlparse("http://x/api/wiki/browse"))
    routes.handle_get(h1, urlparse("http://x/api/wiki/browse?wiki=0"))

    assert h0.get_json() == h1.get_json()


def test_browse_index_one_serves_the_extra_wiki(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse("http://x/api/wiki/browse?wiki=1"))

    assert h.status == 200
    paths = [p["path"] for p in h.get_json()["pages"]]
    assert paths == ["concepts/only-in-extra.md"]


@pytest.mark.parametrize("bad", ["2", "-1", "abc", "1.5", "99999"])
def test_browse_rejects_invalid_index(tmp_path, monkeypatch, bad):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse(f"http://x/api/wiki/browse?wiki={bad}"))

    assert h.status == 400, f"index {bad!r} must be rejected, got {h.status}"


@pytest.mark.parametrize(
    "hostile",
    [
        "/etc",
        "/etc/passwd",
        "../../etc",
        "~/.ssh",
        "%2Fetc",
        "/Users/COLEMAN/.hermes",
    ],
)
def test_wiki_param_can_never_be_a_path(tmp_path, monkeypatch, hostile):
    """The wiki selector is an INDEX. Any path-looking value must 400 rather
    than being interpreted as a filesystem path — this is the whole reason the
    parameter is an index and not a path."""
    from api import routes
    from urllib.parse import urlparse, quote

    _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse(f"http://x/api/wiki/browse?wiki={quote(hostile, safe='')}"))

    assert h.status == 400, f"hostile wiki value {hostile!r} must 400, got {h.status}"
    assert b"root:" not in h.body


def test_page_read_honours_index_and_keeps_wikis_isolated(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)

    # The extra wiki's page is readable via index 1 ...
    h = _FakeHandler()
    routes.handle_get(h, urlparse("http://x/api/wiki/page?wiki=1&path=concepts/only-in-extra.md"))
    assert h.status == 200, h.status
    assert "extra body" in h.get_json()["content"]

    # ... and asking the PRIMARY (index 0) for that path must 404, not fall
    # through to the other wiki.
    h2 = _FakeHandler()
    routes.handle_get(h2, urlparse("http://x/api/wiki/page?wiki=0&path=concepts/only-in-extra.md"))
    assert h2.status == 404, f"cross-wiki path must not resolve, got {h2.status}"


def test_page_read_without_index_uses_primary(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse("http://x/api/wiki/page?path=compounds/psilocybin.md"))

    assert h.status == 200
    assert "Psilocybin" in h.get_json()["content"]


def test_page_read_rejects_invalid_index(tmp_path, monkeypatch):
    from api import routes
    from urllib.parse import urlparse

    _two_wikis(tmp_path, monkeypatch)
    h = _FakeHandler()
    routes.handle_get(h, urlparse("http://x/api/wiki/page?wiki=9&path=compounds/psilocybin.md"))

    assert h.status == 400


def test_indexed_resolver_still_honours_monkeypatched_single_wiki(monkeypatch, tmp_path):
    """The default path must keep going through _llm_wiki_resolve_path so the
    existing single-wiki test contract (monkeypatching that function) holds."""
    from api import routes

    fake = tmp_path / "fake"
    fake.mkdir()
    monkeypatch.setattr(routes, "_llm_wiki_resolve_path", lambda: (fake, "test", True))

    assert routes._llm_wiki_resolve_indexed(None)[0] == fake
    assert routes._llm_wiki_resolve_indexed("")[0] == fake
    assert routes._llm_wiki_resolve_indexed("0")[0] == fake
