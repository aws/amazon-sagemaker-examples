# SPDX-License-Identifier: Apache-2.0
import importlib.util
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "utils/llmperf.py"
spec = importlib.util.spec_from_file_location("llmperf_under_test", SCRIPT)
llmperf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llmperf)


class MonitorTests(unittest.TestCase):
    @patch.object(llmperf, "print")
    def test_drains_output_before_waiting_for_exit(self, printing):
        proc = Mock(pid=42, returncode=0)
        proc.poll.return_value = None
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired("child", 15, output=b"partial"),
            (b"complete output", b"complete errors"),
        ]
        # Bound the baseline regression: its old polling loop must not sleep.
        with patch("time.sleep", side_effect=AssertionError("polling without draining pipes")):
            llmperf.monitor_process(proc)
        self.assertEqual(proc.communicate.call_count, 2)
        proc.communicate.assert_called_with(timeout=15)
        proc.poll.assert_not_called()
        messages = [args[0] for args, _ in printing.call_args_list]
        self.assertEqual(sum("complete output" in message for message in messages), 1)
        self.assertEqual(sum("complete errors" in message for message in messages), 1)
        self.assertFalse(any("partial" in message for message in messages))
        self.assertTrue(any("still running" in message for message in messages))

    @patch.object(llmperf, "print")
    def test_nonzero_exit_and_empty_output(self, printing):
        proc = Mock(pid=42, returncode=7)
        proc.poll.return_value = 7
        proc.communicate.return_value = (b"", b"")
        llmperf.monitor_process(proc)
        printing.assert_called_once_with("Process 42 finished with return code 7")

    @patch.object(llmperf, "print")
    def test_non_utf8_output_does_not_crash_the_monitor(self, printing):
        proc = Mock(pid=42, returncode=0)
        proc.poll.return_value = 0
        proc.communicate.return_value = (b"message\xff", b"error\xff")
        llmperf.monitor_process(proc)
        self.assertTrue(any("message" in args[0] for args, _ in printing.call_args_list))

    @patch.object(llmperf, "print")
    def test_real_verbose_child_completes_without_pipe_deadlock(self, printing):
        code = "import sys; sys.stdout.buffer.write(b'x' * 1048576); sys.stderr.buffer.write(b'y' * 1048576)"
        proc = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        watchdog = threading.Timer(5, proc.kill)
        watchdog.start()
        try:
            # This fails fast on the old polling implementation; cleanup kills the child.
            with patch("time.sleep", side_effect=AssertionError("polling without draining pipes")):
                llmperf.monitor_process(proc)
            self.assertEqual(proc.returncode, 0)
            messages = [args[0] for args, _ in printing.call_args_list]
            self.assertIn("Process output:\n" + "x" * 1048576, messages)
            self.assertIn("Process errors:\n" + "y" * 1048576, messages)
        finally:
            watchdog.cancel()
            if proc.poll() is None:
                proc.kill()
            proc.communicate()


if __name__ == "__main__":
    unittest.main()
