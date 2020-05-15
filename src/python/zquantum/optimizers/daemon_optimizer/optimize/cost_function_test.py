import unittest
import numpy as np
from .cost_function import ProxyCostFunction
from .client_mock import MockedClient
import os


class TestProxyCostFunction(unittest.TestCase):

    def setUp(self):
        self.port = "1234"
        self.ipaddress = "testing-ip"

    def test_evaluate(self):
        # Given
        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        params = np.array([4])
        cost_function = ProxyCostFunction(client)
        target_value = 16

        # When
        value = cost_function.evaluate(params)

        # Then
        self.assertEqual(value, target_value)
        os.remove('client_mock_evaluation_result.json')
        os.remove('current_optimization_params.json')

    def test_callback(self):
        # Given
        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        params = np.array([4])
        cost_function = ProxyCostFunction(client)
        target_value = 16

        # When
        value_1 = cost_function.evaluate(params)
        value_2 = cost_function.evaluate(params)
        cost_function.callback(params)
        history = cost_function.evaluations_history

        # Then
        self.assertEqual(value_1, target_value)
        self.assertEqual(value_2, target_value)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['optimization-evaluation-ids'], ['MOCKED-ID', 'MOCKED-ID', 'MOCKED-ID'])
        np.testing.assert_array_equal(history[0]['params'], params)
        self.assertEqual(history[0]['value'], target_value)
        os.remove('client_mock_evaluation_result.json')
        os.remove('current_optimization_params.json')
