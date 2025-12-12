import sys
import types


class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        pass


if "openai" not in sys.modules:
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=_DummyOpenAI)


class _StubPage:
    def goto(self, *args, **kwargs):
        return None

    def wait_for_timeout(self, *args, **kwargs):
        return None

    def query_selector_all(self, *args, **kwargs):
        return []

    def close(self):
        return None

    def text_content(self, *args, **kwargs):
        return ""


class _StubContext:
    def new_page(self):
        return _StubPage()

    def close(self):
        return None


class _StubBrowser:
    def new_page(self):
        return _StubPage()

    def new_context(self):
        return _StubContext()

    def close(self):
        return None


class _StubChromium:
    def launch(self, headless=True):
        return _StubBrowser()


class _StubPlaywright:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @property
    def chromium(self):
        return _StubChromium()


def _sync_playwright():
    return _StubPlaywright()


if "playwright" not in sys.modules:
    sync_api_stub = types.SimpleNamespace(sync_playwright=_sync_playwright, TimeoutError=RuntimeError)
    sys.modules["playwright"] = types.SimpleNamespace(sync_api=sync_api_stub)
    sys.modules["playwright.sync_api"] = sync_api_stub
