#!/usr/bin/env python3
import sys
import unittest
from unittest import mock

sys.path.insert(0, "../")

import helpers


class GeneratePostTest(unittest.TestCase):
    def test_success_first_try(self):
        with mock.patch.object(helpers, "post", return_value={"ok": 1}) as p:
            self.assertEqual(helpers.generate_post("generate", {"prompt": "x"}), {"ok": 1})
            p.assert_called_once()

    def test_non_connection_error_raises_immediately(self):
        # HTTP 500 / non-transient error: not retried, surfaced immediately.
        with mock.patch.object(helpers, "post",
                               side_effect=RuntimeError("HTTP 500: boom")) as p, \
             mock.patch.object(helpers.time, "sleep") as s:
            with self.assertRaises(RuntimeError):
                helpers.generate_post("generate", {})
            p.assert_called_once()
            s.assert_not_called()

    def test_connection_error_retries_with_backoff_same_server(self):
        # Connection dropped: retry against the SAME server (model stays loaded).
        calls = {"n": 0}
        def flaky(e, payload):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("Connection aborted by remote")
            return {"ok": 1}
        with mock.patch.object(helpers, "post", side_effect=flaky) as p, \
             mock.patch.object(helpers.time, "sleep") as s:
            self.assertEqual(helpers.generate_post("generate", {}, attempts=3), {"ok": 1})
            self.assertEqual(p.call_count, 3)
            # backoff 2s then 4s
            s.assert_has_calls([mock.call(2.0), mock.call(4.0)])

    def test_exhausts_retries_raises_last_error(self):
        def always_fail(e, payload):
            raise ConnectionError("Connection aborted by remote")
        with mock.patch.object(helpers, "post", side_effect=always_fail) as p, \
             mock.patch.object(helpers.time, "sleep") as s:
            with self.assertRaises(ConnectionError):
                helpers.generate_post("generate", {}, attempts=3)
            self.assertEqual(p.call_count, 3)


class PingWithRetriesTest(unittest.TestCase):
    def test_returns_first_success(self):
        with mock.patch.object(helpers, "ping_server",
                               side_effect=[None, None, {"status": "ok"}]) as p, \
             mock.patch.object(helpers.time, "sleep"):
            self.assertEqual(helpers.ping_with_retries(3, 1.0), {"status": "ok"})
            self.assertEqual(p.call_count, 3)

    def test_returns_none_when_never_up(self):
        with mock.patch.object(helpers, "ping_server", return_value=None) as p, \
             mock.patch.object(helpers.time, "sleep"):
            self.assertIsNone(helpers.ping_with_retries(2, 1.0))
            self.assertEqual(p.call_count, 2)


class ConnectRefusedTest(unittest.TestCase):
    def test_refused_reported_true(self):
        def refused():
            raise ConnectionError("Connection refused by remote host")
        with mock.patch.object(helpers.requests, "get", side_effect=refused):
            self.assertTrue(helpers._connect_refused())

    def test_timeout_reported_false(self):
        # A read-timeout means busy (loading), not dead.
        def timeout(*a, **k):
            raise requests_exceptions_timeout()
        class requests_exceptions_timeout(Exception):
            pass
        with mock.patch.object(helpers.requests, "get", side_effect=timeout):
            self.assertFalse(helpers._connect_refused())


class EnsureServerTest(unittest.TestCase):
    def test_returns_ping_when_alive_no_restart(self):
        with mock.patch.object(helpers, "ping_with_retries",
                               return_value={"status": "ok"}) as p, \
             mock.patch.object(helpers, "stop_service") as stop, \
             mock.patch.object(helpers, "start_service") as start:
            self.assertEqual(helpers.ensure_server("d"), {"status": "ok"})
            p.assert_called_once()
            stop.assert_not_called()
            start.assert_not_called()

    def test_does_not_restart_busy_server(self):
        # ping times out (busy loading) but connection NOT refused: never restart.
        with mock.patch.object(helpers, "ping_with_retries", return_value=None), \
             mock.patch.object(helpers, "_connect_refused", return_value=False), \
             mock.patch.object(helpers, "stop_service") as stop, \
             mock.patch.object(helpers, "start_service") as start:
            self.assertIsNone(helpers.ensure_server("d"))
            stop.assert_not_called()
            start.assert_not_called()

    def test_recovers_when_server_down(self):
        # Probes fail AND connection refused (process gone) -> restart -> re-ping.
        with mock.patch.object(helpers, "ping_with_retries",
                               side_effect=[None, {"status": "ok"}]), \
             mock.patch.object(helpers, "_connect_refused", return_value=True), \
             mock.patch.object(helpers, "stop_service") as stop, \
             mock.patch.object(helpers, "start_service") as start, \
             mock.patch.object(helpers, "adb_forward"), \
             mock.patch.object(helpers.time, "sleep"):
            self.assertEqual(helpers.ensure_server("d"), {"status": "ok"})
            stop.assert_called_once()
            start.assert_called_once()

    def test_returns_none_after_failed_recovery(self):
        # Server never comes back: ensure_server reports None honestly.
        with mock.patch.object(helpers, "ping_with_retries", return_value=None), \
             mock.patch.object(helpers, "_connect_refused", return_value=True), \
             mock.patch.object(helpers, "stop_service"), \
             mock.patch.object(helpers, "start_service"), \
             mock.patch.object(helpers, "adb_forward"), \
             mock.patch.object(helpers.time, "sleep"):
            self.assertIsNone(helpers.ensure_server("d"))


if __name__ == "__main__":
    unittest.main()
