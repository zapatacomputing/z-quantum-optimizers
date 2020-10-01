import unittest
import numpy as np
import os
from .cost_function import ProxyCostFunction
from .client_mock import MockedClient
from zquantum.core.utils import ValueEstimate


class TestProxyCostFunction(unittest.TestCase):
    def setUp(self):
        self.port = "1234"
        self.ipaddress = "testing-ip"

        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        self.cost_functions = [ProxyCostFunction(client)]
        self.params_sizes = [4]

    def test_evaluate(self):
        # Given
        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        params = np.array([4])
        cost_function = ProxyCostFunction(client)
        target_value = ValueEstimate(16)

        # When
        value = cost_function(params)

        # Then
        self.assertEqual(value, target_value)
        os.remove("client_mock_evaluation_result.json")
        os.remove("current_optimization_params.json")

    def test_callback(self):
        # Given
        client = MockedClient(self.ipaddress, self.port, "return_x_squared")
        params = np.array([4])
        cost_function = ProxyCostFunction(client)
        target_value = ValueEstimate(16)

        # When
        value_1 = cost_function(params)
        value_2 = cost_function(params)
        cost_function.callback(params)

        # Then
        self.assertEqual(value_1, target_value)
        self.assertEqual(value_2, target_value)

        os.remove("client_mock_evaluation_result.json")
        os.remove("current_optimization_params.json")
