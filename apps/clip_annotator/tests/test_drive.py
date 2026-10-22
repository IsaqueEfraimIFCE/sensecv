"""DriveClient retry behaviour, against a fake googleapiclient service."""
import json

import httplib2
import pytest
from googleapiclient.errors import HttpError

from clip_annotator import drive as drive_module
from clip_annotator.drive import DriveClient, DriveQuotaError


class _Request:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def execute(self):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Service:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0

    def files(self):
        return self

    def list(self, **kwargs):
        self.calls += 1
        return _Request(self.outcomes)


def _http_error(status, reason=""):
    body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
    return HttpError(httplib2.Response({"status": status}), body)


def _client(outcomes):
    client = DriveClient(credentials=None)
    service = _Service(list(outcomes))
    client._local.service = service  # bypass building a real service
    return client, service


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(drive_module.time, "sleep", lambda seconds: None)


def test_network_timeouts_are_retried():
    client, service = _client(
        [TimeoutError("read timed out"), ConnectionResetError(), {"files": [{"id": "abc"}]}]
    )

    assert client.find("x.jpg", "parent") == "abc"
    assert service.calls == 3


def test_server_errors_are_retried():
    client, service = _client([_http_error(503), {"files": [{"id": "abc"}]}])

    assert client.find("x.jpg", "parent") == "abc"
    assert service.calls == 2


def test_gives_up_after_the_attempt_limit():
    client, service = _client([TimeoutError("t")] * drive_module.MAX_ATTEMPTS)

    with pytest.raises(TimeoutError):
        client.find("x.jpg", "parent")
    assert service.calls == drive_module.MAX_ATTEMPTS


def test_client_errors_are_not_retried():
    client, service = _client([_http_error(404)])

    with pytest.raises(HttpError):
        client.find("x.jpg", "parent")
    assert service.calls == 1


def test_storage_quota_is_reported_with_an_actionable_error():
    client, service = _client([_http_error(403, "storageQuotaExceeded")])

    with pytest.raises(DriveQuotaError):
        client.find("x.jpg", "parent")
    assert service.calls == 1


def test_programming_errors_are_not_swallowed():
    client, service = _client([ValueError("bug")])

    with pytest.raises(ValueError):
        client.find("x.jpg", "parent")
    assert service.calls == 1
