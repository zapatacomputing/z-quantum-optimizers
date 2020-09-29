# NOTE: Mock Server Tests from: https://gist.github.com/eruvanos/f6f62edb368a20aaa880e12976620db8

import unittest
import requests
import socket
import time

from .server_mock import MockServer


class TestMockServer(unittest.TestCase):
    def setUp(self):
        self.server = MockServer(port=8888)
        self.server.start()

        # Get the proxy IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("localhost", 80))
        self.ipaddress = str(s.getsockname()[0])
        s.close()

        self.max_tries = 60
        counter = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            while s.connect_ex((self.ipaddress, 8888)) != 0:
                time.sleep(1)
                counter += 1
                if counter > self.max_tries:
                    raise SystemExit("Testing server took too long to start.")

    def test_mock_with_callback(self):
        self.called = False

        def callback():
            self.called = True
            return "Hallo"

        self.server.add_callback_response("/callback", callback)

        response = requests.get(self.server.url + "/callback")

        self.assertEqual(200, response.status_code)
        self.assertEqual("Hallo", response.text)
        self.assertTrue(self.called)

    def tearDown(self):
        self.server.shutdown_server()

        counter = 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            while s.connect_ex((self.ipaddress, 8888)) == 0:
                time.sleep(1)
                counter += 1
                if counter > self.max_tries:
                    raise SystemExit("Testing server took too long to start.")
