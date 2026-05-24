"""
Root test conftest — stubs out heavy/unavailable dependencies so that
services which import from `database.supabase_client` can be collected
without requiring the real `python-dotenv` and `supabase` packages to
be installed in the test environment.

Individual tests patch `services.sentiment.*.supabase` (or similar) as
needed; this stub just prevents ImportError at collection time.
"""
import sys
import types
from unittest.mock import MagicMock


def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules.setdefault(name, mod)
    return mod


# Stub python-dotenv
_dotenv = _stub_module("dotenv", load_dotenv=lambda *a, **kw: None)

# Stub supabase package
_supa_mock = MagicMock()
_stub_module("supabase", create_client=lambda url, key: _supa_mock)
