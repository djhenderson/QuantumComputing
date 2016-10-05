#! python
# -*- coding:utf-8 -*-
# flake8 --ignore=E501,N802 tests.py

from __future__ import division, print_function

import itertools
from math import sqrt
import numpy as np
import time
import unittest

from simulation import Gate, Probability, Programs, QuantumComputer, QuantumRegister, State


def print_timing(name, start):
    print("\n%.3f %s\t" % ((time.time() - start), name), end='')

#########################################################################################
# All test code below
#########################################################################################


class TestQuantumRegister(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.q0 = QuantumRegister("q0")
        self.q1 = QuantumRegister("q1")

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)
        self.q0 = None
        self.q1 = None

    def test_get_num_qubits(self):
        self.assertTrue(self.q0.get_num_qubits() == self.q0.get_num_qubits() == 1)

    def test_equality(self):
        self.assertEqual(self.q0, self.q0)
        self.assertNotEqual(self.q0, self.q1)


class TestMeasure(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def test_measure_probs_plus(self):
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.plus_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.5)), rtol=1e-2))

    def test_measure_probs_minus(self):
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.minus_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.5)), rtol=1e-2))

    def test_collapse(self):
        result = None
        for i in range(100):
            if result is None:
                result = State.measure(State.minus_state)
            else:
                new_measure = State.measure(result)
                self.assertTrue(np.allclose(result, new_measure))
                result = new_measure

    def test_measure_bell(self):
        """ Tests the measurement of a 2 qubit entangled system"""
        measurements = []
        for i in range(100000):
            measurements += [State.measure(State.bell_state)]
        result = (1. * sum(measurements)) / len(measurements)
        self.assertTrue(np.allclose(list(result.flat), np.array((0.5, 0.0, 0.0, 0.5)), rtol=1e-2))


class TestGetBloch(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def test_get_bloch(self):
        self.assertTrue(np.allclose(State.get_bloch(State.zero_state), np.array((0, 0, 1))))
        self.assertTrue(np.allclose(State.get_bloch(State.one_state), np.array((0, 0, -1))))
        self.assertTrue(np.allclose(State.get_bloch(State.plusi_state), np.array((0, 1, 0))))
        self.assertTrue(np.allclose(State.get_bloch(State.minusi_state), np.array((0, -1, 0))))
        self.assertTrue(np.allclose(State.get_bloch(Gate.Z * State.plus_state), np.array((-1, 0, 0))))
        self.assertTrue(np.allclose(State.get_bloch(Gate.Z * State.minus_state), np.array((1, 0, 0))))

        # assert the norms are 1 for cardinal points (obviously) but also for a few other points at higher T depth on the Bloch Sphere
        for state in [State.zero_state, State.one_state, State.plusi_state, State.minusi_state, Gate.Z * State.plus_state, Gate.H * Gate.T * Gate.Z * State.plus_state, Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.Z * State.plus_state]:
            self.assertAlmostEqual(np.linalg.norm(state), 1.0)


class TestGetBloch2(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def get_bloch_2(self, state):
        """ equal to get_bloch just a different way of calculating things. Used for testing get_bloch. """
        return np.array((((state * state.conjugate().transpose() * Gate.X).trace()).item(0), ((state * state.conjugate().transpose() * Gate.Y).trace()).item(0), ((state * state.conjugate().transpose() * Gate.Z).trace()).item(0)))

    def test_get_bloch_2(self):
        self.assertTrue(np.allclose(self.get_bloch_2(State.zero_state), State.get_bloch(State.zero_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.one_state), State.get_bloch(State.one_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.plusi_state), State.get_bloch(State.plusi_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(State.minusi_state), State.get_bloch(State.minusi_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(Gate.Z * State.plus_state), State.get_bloch(Gate.Z * State.plus_state)))
        self.assertTrue(np.allclose(self.get_bloch_2(Gate.H * Gate.T * Gate.Z * State.plus_state), State.get_bloch(Gate.H * Gate.T * Gate.Z * State.plus_state)))  # test for arbitrary gates


class TestCNOTGate(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def test_CNOT(self):
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('00'), State.state_from_string('00')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('01'), State.state_from_string('01')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('10'), State.state_from_string('11')))
        self.assertTrue(np.allclose(Gate.CNOT2_01 * State.state_from_string('11'), State.state_from_string('10')))


class TestTGate(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def test_T(self):
        # This is useful to check some of the exercises on IBM's quantum experience.
        # "Ground truth" answers from IBM's calculations which unfortunately are not reported to high precision.
        red_state = Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        green_state = Gate.S * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        blue_state = Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state

        self.assertTrue(np.allclose(State.get_bloch(red_state), np.array((0.5, 0.5, 0.707)), rtol=1e-3))
        self.assertTrue(np.allclose(State.get_bloch(green_state), np.array((0.427, 0.457, 0.780)), rtol=1e-3))
        self.assertTrue(np.allclose(State.get_bloch(blue_state), np.array((0.457, 0.427, 0.780)), rtol=1e-3))

        # Checking norms
        for state in [red_state, green_state, blue_state]:
            self.assertAlmostEqual(np.linalg.norm(state), 1.0)


class TestMultiQuantumRegisterStates(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        # Two qubit states (basis)
        # To derive the ordering you do ((+) is outer product):
        # Symbolically: |00> = |0> (+) |0>; gives 4x1
        # In Python: np.kron(zero_state, zero_state)
        self.one_qubits_0 = np.matrix('1; 0')
        self.one_qubits_1 = np.matrix('0; 1')

        self.two_qubits_00 = np.kron(State.zero_state, State.zero_state)
        self.two_qubits_01 = np.kron(State.zero_state, State.one_state)
        self.two_qubits_10 = np.kron(State.one_state, State.zero_state)
        self.two_qubits_11 = np.kron(State.one_state, State.one_state)

        # Three qubit states (basis)
        self.three_qubits_000 = np.kron(self.two_qubits_00, State.zero_state)
        self.three_qubits_001 = np.kron(self.two_qubits_00, State.one_state)
        self.three_qubits_010 = np.kron(self.two_qubits_01, State.zero_state)
        self.three_qubits_011 = np.kron(self.two_qubits_01, State.one_state)
        self.three_qubits_100 = np.kron(self.two_qubits_10, State.zero_state)
        self.three_qubits_101 = np.kron(self.two_qubits_10, State.one_state)
        self.three_qubits_110 = np.kron(self.two_qubits_11, State.zero_state)
        self.three_qubits_111 = np.kron(self.two_qubits_11, State.one_state)

        # Four qubit states (basis)
        self.four_qubits_0000 = np.kron(self.three_qubits_000, State.zero_state)
        self.four_qubits_0001 = np.kron(self.three_qubits_000, State.one_state)
        self.four_qubits_0010 = np.kron(self.three_qubits_001, State.zero_state)
        self.four_qubits_0011 = np.kron(self.three_qubits_001, State.one_state)
        self.four_qubits_0100 = np.kron(self.three_qubits_010, State.zero_state)
        self.four_qubits_0101 = np.kron(self.three_qubits_010, State.one_state)
        self.four_qubits_0110 = np.kron(self.three_qubits_011, State.zero_state)
        self.four_qubits_0111 = np.kron(self.three_qubits_011, State.one_state)
        self.four_qubits_1000 = np.kron(self.three_qubits_100, State.zero_state)
        self.four_qubits_1001 = np.kron(self.three_qubits_100, State.one_state)
        self.four_qubits_1010 = np.kron(self.three_qubits_101, State.zero_state)
        self.four_qubits_1011 = np.kron(self.three_qubits_101, State.one_state)
        self.four_qubits_1100 = np.kron(self.three_qubits_110, State.zero_state)
        self.four_qubits_1101 = np.kron(self.three_qubits_110, State.one_state)
        self.four_qubits_1110 = np.kron(self.three_qubits_111, State.zero_state)
        self.four_qubits_1111 = np.kron(self.three_qubits_111, State.one_state)

        # Five qubit states (basis)
        self.five_qubits_00000 = np.kron(self.four_qubits_0000, State.zero_state)
        self.five_qubits_00001 = np.kron(self.four_qubits_0000, State.one_state)
        self.five_qubits_00010 = np.kron(self.four_qubits_0001, State.zero_state)
        self.five_qubits_00011 = np.kron(self.four_qubits_0001, State.one_state)
        self.five_qubits_00100 = np.kron(self.four_qubits_0010, State.zero_state)
        self.five_qubits_00101 = np.kron(self.four_qubits_0010, State.one_state)
        self.five_qubits_00110 = np.kron(self.four_qubits_0011, State.zero_state)
        self.five_qubits_00111 = np.kron(self.four_qubits_0011, State.one_state)
        self.five_qubits_01000 = np.kron(self.four_qubits_0100, State.zero_state)
        self.five_qubits_01001 = np.kron(self.four_qubits_0100, State.one_state)
        self.five_qubits_01010 = np.kron(self.four_qubits_0101, State.zero_state)
        self.five_qubits_01011 = np.kron(self.four_qubits_0101, State.one_state)
        self.five_qubits_01100 = np.kron(self.four_qubits_0110, State.zero_state)
        self.five_qubits_01101 = np.kron(self.four_qubits_0110, State.one_state)
        self.five_qubits_01110 = np.kron(self.four_qubits_0111, State.zero_state)
        self.five_qubits_01111 = np.kron(self.four_qubits_0111, State.one_state)
        self.five_qubits_10000 = np.kron(self.four_qubits_1000, State.zero_state)
        self.five_qubits_10001 = np.kron(self.four_qubits_1000, State.one_state)
        self.five_qubits_10010 = np.kron(self.four_qubits_1001, State.zero_state)
        self.five_qubits_10011 = np.kron(self.four_qubits_1001, State.one_state)
        self.five_qubits_10100 = np.kron(self.four_qubits_1010, State.zero_state)
        self.five_qubits_10101 = np.kron(self.four_qubits_1010, State.one_state)
        self.five_qubits_10110 = np.kron(self.four_qubits_1011, State.zero_state)
        self.five_qubits_10111 = np.kron(self.four_qubits_1011, State.one_state)
        self.five_qubits_11000 = np.kron(self.four_qubits_1100, State.zero_state)
        self.five_qubits_11001 = np.kron(self.four_qubits_1100, State.one_state)
        self.five_qubits_11010 = np.kron(self.four_qubits_1101, State.zero_state)
        self.five_qubits_11011 = np.kron(self.four_qubits_1101, State.one_state)
        self.five_qubits_11100 = np.kron(self.four_qubits_1110, State.zero_state)
        self.five_qubits_11101 = np.kron(self.four_qubits_1110, State.one_state)
        self.five_qubits_11110 = np.kron(self.four_qubits_1111, State.zero_state)
        self.five_qubits_11111 = np.kron(self.four_qubits_1111, State.one_state)

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)

    def test_basis(self):
        # # Sanity checks

        # 1-qubit
        self.assertTrue(np.allclose(State.zero_state + State.one_state, np.matrix('1; 1')))
        eye = np.eye(2, 2)
        for row, state in enumerate([State.zero_state, State.one_state]):
            self.assertTrue(np.allclose(state.transpose(), eye[row]))

        # 2-qubit
        self.assertTrue(np.allclose(self.two_qubits_00 + self.two_qubits_01 + self.two_qubits_10 + self.two_qubits_11, np.matrix('1; 1; 1; 1')))
        eye = np.eye(4, 4)
        for row, state in enumerate([self.two_qubits_00, self.two_qubits_01, self.two_qubits_10, self.two_qubits_11]):
            self.assertTrue(np.allclose(state.transpose(), eye[row]))

        # 3-qubit
        self.assertTrue(np.allclose(self.three_qubits_000 + self.three_qubits_001 + self.three_qubits_010 + self.three_qubits_011 + self.three_qubits_100 + self.three_qubits_101 + self.three_qubits_110 + self.three_qubits_111, np.matrix('1; 1; 1; 1; 1; 1; 1; 1')))
        eye = np.eye(8, 8)
        for row, state in enumerate([self.three_qubits_000, self.three_qubits_001, self.three_qubits_010, self.three_qubits_011, self.three_qubits_100, self.three_qubits_101, self.three_qubits_110, self.three_qubits_111]):
            self.assertTrue(np.allclose(state.transpose(), eye[row]))

        # 4-qubit
        self.assertTrue(np.allclose(self.four_qubits_0000 + self.four_qubits_0001 + self.four_qubits_0010 + self.four_qubits_0011 + self.four_qubits_0100 + self.four_qubits_0101 + self.four_qubits_0110 + self.four_qubits_0111 + self.four_qubits_1000 + self.four_qubits_1001 + self.four_qubits_1010 + self.four_qubits_1011 + self.four_qubits_1100 + self.four_qubits_1101 + self.four_qubits_1110 + self.four_qubits_1111, np.matrix('1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1')))
        eye = np.eye(16, 16)
        for row, state in enumerate([self.four_qubits_0000, self.four_qubits_0001, self.four_qubits_0010, self.four_qubits_0011, self.four_qubits_0100, self.four_qubits_0101, self.four_qubits_0110, self.four_qubits_0111, self.four_qubits_1000, self.four_qubits_1001, self.four_qubits_1010, self.four_qubits_1011, self.four_qubits_1100, self.four_qubits_1101, self.four_qubits_1110, self.four_qubits_1111]):
            self.assertTrue(np.allclose(state.transpose(), eye[row]))

        # 5-qubit
        self.assertTrue(np.allclose(self.five_qubits_00000 + self.five_qubits_00001 + self.five_qubits_00010 + self.five_qubits_00011 + self.five_qubits_00100 + self.five_qubits_00101 + self.five_qubits_00110 + self.five_qubits_00111 + self.five_qubits_01000 + self.five_qubits_01001 + self.five_qubits_01010 + self.five_qubits_01011 + self.five_qubits_01100 + self.five_qubits_01101 + self.five_qubits_01110 + self.five_qubits_01111 + self.five_qubits_10000 + self.five_qubits_10001 + self.five_qubits_10010 + self.five_qubits_10011 + self.five_qubits_10100 + self.five_qubits_10101 + self.five_qubits_10110 + self.five_qubits_10111 + self.five_qubits_11000 + self.five_qubits_11001 + self.five_qubits_11010 + self.five_qubits_11011 + self.five_qubits_11100 + self.five_qubits_11101 + self.five_qubits_11110 + self.five_qubits_11111, np.matrix('1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1')))
        eye = np.eye(32, 32)
        for row, state in enumerate([self.five_qubits_00000, self.five_qubits_00001, self.five_qubits_00010, self.five_qubits_00011, self.five_qubits_00100, self.five_qubits_00101, self.five_qubits_00110, self.five_qubits_00111, self.five_qubits_01000, self.five_qubits_01001, self.five_qubits_01010, self.five_qubits_01011, self.five_qubits_01100, self.five_qubits_01101, self.five_qubits_01110, self.five_qubits_01111, self.five_qubits_10000, self.five_qubits_10001, self.five_qubits_10010, self.five_qubits_10011, self.five_qubits_10100, self.five_qubits_10101, self.five_qubits_10110, self.five_qubits_10111, self.five_qubits_11000, self.five_qubits_11001, self.five_qubits_11010, self.five_qubits_11011, self.five_qubits_11100, self.five_qubits_11101, self.five_qubits_11110, self.five_qubits_11111]):
            self.assertTrue(np.allclose(state.transpose(), eye[row]))

    def test_separate_state(self):
        value_groups = \
            [
                State.separate_state(self.five_qubits_11010),
                State.separate_state(self.four_qubits_0101),
                State.separate_state(self.three_qubits_000),
                State.separate_state(self.three_qubits_111),
                State.separate_state(self.three_qubits_101),
                State.separate_state(self.two_qubits_00),
                State.separate_state(self.two_qubits_01),
                State.separate_state(self.two_qubits_10),
                State.separate_state(self.two_qubits_11),
                # State.separate_state(self.one_qubits_0),
                # State.separate_state(self.one_qubits_1)
            ]

        target_groups = \
            [
                (State.one_state, State.one_state, State.zero_state, State.one_state, State.zero_state),
                (State.zero_state, State.one_state, State.zero_state, State.one_state),
                (State.zero_state, State.zero_state, State.zero_state),
                (State.one_state, State.one_state, State.one_state),
                (State.one_state, State.zero_state, State.one_state),
                (State.zero_state, State.zero_state),
                (State.zero_state, State.one_state),
                (State.one_state, State.zero_state),
                (State.one_state, State.one_state),
                # (State.zero_state),
                # (State.one_state)
            ]
        def clean(s):  # DEBUG
            # print("DEBUG: s:", repr(s))
            s = s.replace('\r', '')
            s = s.replace('\n', '')
            s = s.replace('\t', ' ')
            s = s.replace('        ', ' ')
            # print("DEBUG: s:", repr(s))
            return s
        for vg, tg in zip(value_groups, target_groups):
            # print("DEBUG:\tvg:", clean(str(vg)), "\n\ttg:", clean(str(tg)))
            # for value_state, target_state in zip(value_groups, target_groups):
            for value_state, target_state in zip(vg, tg):
                # print("DEBUG:\tvalue_state :", clean(repr(value_state)), "\n\ttarget_state:", clean(repr(target_state)))
                self.assertTrue(np.allclose(value_state, target_state))

    def test_string_from_state(self):
        self.assertEqual(State.string_from_state(State.zero_state), '0')
        self.assertEqual(State.string_from_state(State.one_state), '1')
        self.assertEqual(State.string_from_state(self.two_qubits_00), '00')
        self.assertEqual(State.string_from_state(self.two_qubits_01), '01')
        self.assertEqual(State.string_from_state(self.two_qubits_10), '10')
        self.assertEqual(State.string_from_state(self.two_qubits_11), '11')
        self.assertEqual(State.string_from_state(self.three_qubits_110), '110')
        self.assertEqual(State.string_from_state(self.four_qubits_1101), '1101')
        self.assertEqual(State.string_from_state(self.five_qubits_11010), '11010')

    def test_state_from_string(self):
        for value_group, target_group in \
                zip(
                    ['0', '1', '00', '01', '10', '11', '110', '1101', '11010'],
                    [
                        [State.zero_state],
                        [State.one_state],
                        [State.zero_state, State.zero_state],
                        [State.zero_state, State.one_state],
                        [State.one_state, State.zero_state],
                        [State.one_state, State.one_state],
                        [State.one_state, State.one_state, State.zero_state],
                        [State.one_state, State.one_state, State.zero_state, State.one_state],
                        [State.one_state, State.one_state, State.zero_state, State.one_state, State.zero_state]
                    ]
                ):
            self.assertEqual(value_group, State.string_from_state(State.state_from_string(value_group)))
            value_group = State.separate_state(State.state_from_string(value_group))
            self.assertEqual(len(value_group), len(target_group))
            for value_state, target_state in zip(value_group, target_group):
                self.assertTrue(np.allclose(value_state, target_state))


class TestQuantumComputer(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.qc = QuantumComputer()

    def tearDown(self):
        print_timing(self._testMethodName, self.startTime)
        self.qc = None

    def test_apply_gate(self):
        self.qc.apply_gate(Gate.H * Gate.T * Gate.Sdagger * Gate.Tdagger * Gate.X * Gate.Y, "q0")
        self.assertTrue(self.qc.qubit_states_equal("q0", Gate.H * Gate.T * Gate.Sdagger * Gate.Tdagger * Gate.X * Gate.Y * State.zero_state))
        # Some tests on entangled gates, breaking abstraction but will improve testing soon
        self.qc.reset()
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q0.set_state(np.kron(State.zero_state, State.zero_state))
        self.qc.qubits.entangle_quantum_registers(q0, q1)

        # We will test applying the gate to qubits one and two
        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '10')
        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '00')

        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q1").name, "q1")
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '01')
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '00')
        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '10')
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '11')

        # Now testing on 3 qubits
        q3 = self.qc.qubits.get_quantum_register_containing("q3")
        q0.set_state(np.kron(np.kron(State.zero_state, State.zero_state), State.zero_state))
        self.qc.qubits.entangle_quantum_registers(q0, q3)
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q1").name, "q1")
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q3").name, "q3")
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q0").name, "q0")
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q2").name, "q2")
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q4").name, "q4")

        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '100')
        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '000')
        self.assertEqual(self.qc.qubits.get_quantum_register_containing("q1").name, "q1")
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '010')
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '000')
        self.qc.apply_gate(Gate.X, "q0")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '100')
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '110')
        self.qc.apply_gate(Gate.X, "q3")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '111')
        self.qc.apply_gate(Gate.X, "q1")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q0").get_state()), '101')
        self.qc.apply_gate(Gate.X, "q4")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q4").get_state()), '1')
        self.qc.apply_gate(Gate.X, "q4")
        self.assertEqual(State.string_from_state(self.qc.qubits.get_quantum_register_containing("q4").get_state()), '0')

    def test_apply_two_qubit_gate_CNOT_target(self):
        self.assertTrue(self.qc.qubit_states_equal("q0", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.zero_state))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")
        self.assertTrue(self.qc.qubit_states_equal("q0", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.zero_state))
        self.qc.apply_gate(Gate.X, "q0")
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")
        self.assertTrue(self.qc.qubit_states_equal("q0", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.one_state))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")
        self.assertTrue(self.qc.qubit_states_equal("q0", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.zero_state))

    def test_apply_two_qubit_gate_CNOT_two_entangled_target(self):
        # We'll put qubit0 in state |10> and qubit1 is in state |0>
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q0.set_state(State.state_from_string("10"))
        self.qc.qubits.entangle_quantum_registers(q0, q1)
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 100 After: 101
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('101')))
        self.qc.reset()
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q0.set_state(State.state_from_string("10"))
        self.qc.qubits.entangle_quantum_registers(q0, q1)
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 100 After: 100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 100 After: 110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('110')))

    def test_apply_two_qubit_gate_CNOT_three_entangled_target(self):
        # Entangled already
        # Put q0 in an entangled state: |000>
        for target, control in list(itertools.product(["q0", "q1", "q2"], repeat=2)):
            if target != control:
                self.qc.reset()
                q0 = self.qc.qubits.get_quantum_register_containing("q0")
                q1 = self.qc.qubits.get_quantum_register_containing("q1")
                q2 = self.qc.qubits.get_quantum_register_containing("q2")
                q0.set_state(State.state_from_string("000"))
                self.qc.qubits.entangle_quantum_registers(q0, q1)
                self.qc.qubits.entangle_quantum_registers(q0, q2)
                self.assertEqual(QuantumRegister.num_qubits(q0.get_state()), 3)
                self.qc.apply_two_qubit_gate_CNOT(target, control)
                self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('000')))
        self.qc.reset()
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q2 = self.qc.qubits.get_quantum_register_containing("q2")
        q0.set_state(State.state_from_string("100"))
        self.qc.qubits.entangle_quantum_registers(q0, q1)
        self.qc.qubits.entangle_quantum_registers(q0, q2)
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 100 After: 110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('110')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 110 After: 010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('010')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 010 After: 010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('010')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 010 After: 010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('010')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 010 After: 011
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('011')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 011 After: 001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('001')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 001 After: 001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('001')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 001 After: 001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('001')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 001 After: 001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('001')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 001 After: 001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('001')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 001 After: 011
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('011')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 011 After: 111
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('111')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 111 After: 110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('110')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 110 After: 110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('110')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 110 After: 110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2", State.state_from_string('110')))

    def test_apply_two_qubit_gate_CNOT_four_entangled_target(self):
        # Entangled already
        # Put q0 in an entangled state: |0000>
        for target, control in list(itertools.product(["q0", "q1", "q2", "q3"], repeat=2)):
            if target != control:
                self.qc.reset()
                q0 = self.qc.qubits.get_quantum_register_containing("q0")
                q1 = self.qc.qubits.get_quantum_register_containing("q1")
                q2 = self.qc.qubits.get_quantum_register_containing("q2")
                q3 = self.qc.qubits.get_quantum_register_containing("q3")
                q0.set_state(State.state_from_string("0000"))
                self.qc.qubits.entangle_quantum_registers(q0, q1)
                self.qc.qubits.entangle_quantum_registers(q0, q2)
                self.qc.qubits.entangle_quantum_registers(q0, q3)

                self.assertEqual(QuantumRegister.num_qubits(q0.get_state()), 4)
                self.qc.apply_two_qubit_gate_CNOT(target, control)
                self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0000')))
        self.qc.reset()
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q2 = self.qc.qubits.get_quantum_register_containing("q2")
        q3 = self.qc.qubits.get_quantum_register_containing("q3")
        q0.set_state(State.state_from_string("1000"))
        self.qc.qubits.entangle_quantum_registers(q0, q1)
        self.qc.qubits.entangle_quantum_registers(q0, q2)
        self.qc.qubits.entangle_quantum_registers(q0, q3)

        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 1000 After: 1100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1100')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 1100 After: 0100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 0100 After: 0100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 0100 After: 0100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0100')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 0100 After: 0110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0110')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 0110 After: 0010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0010')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 0010 After: 0010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0010')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 0010 After: 0010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0010')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 0010 After: 0010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0010')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 0010 After: 0010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0010')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 0010 After: 0110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('0110')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 0110 After: 1110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1110')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 1110 After: 1100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 1100 After: 1100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 1100 After: 1100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q3")  # Before: 1100 After: 1101
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1101')))
        self.qc.apply_two_qubit_gate_CNOT("q3", "q2")  # Before: 1101 After: 1111
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1111')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q3")  # Before: 1111 After: 1110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1110')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 1110 After: 1010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1010')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q3")  # Before: 1010 After: 1010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1010')))
        self.qc.apply_two_qubit_gate_CNOT("q3", "q1")  # Before: 1010 After: 1010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3", State.state_from_string('1010')))

    def test_apply_two_qubit_gate_CNOT_five_entangled_target(self):
        # Entangled already
        # Put q0 in an entangled state: |00000>
        for target, control in list(itertools.product(["q0", "q1", "q2", "q3", "q4"], repeat=2)):
            if target != control:
                self.qc.reset()
                q0 = self.qc.qubits.get_quantum_register_containing("q0")
                q1 = self.qc.qubits.get_quantum_register_containing("q1")
                q2 = self.qc.qubits.get_quantum_register_containing("q2")
                q3 = self.qc.qubits.get_quantum_register_containing("q3")
                q4 = self.qc.qubits.get_quantum_register_containing("q4")
                q0.set_state(State.state_from_string("00000"))
                self.qc.qubits.entangle_quantum_registers(q0, q1)
                self.qc.qubits.entangle_quantum_registers(q0, q2)
                self.qc.qubits.entangle_quantum_registers(q0, q3)
                self.qc.qubits.entangle_quantum_registers(q0, q4)
                self.assertEqual(QuantumRegister.num_qubits(q0.get_state()), 5)
                self.qc.apply_two_qubit_gate_CNOT(target, control)
                self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00000')))
        self.qc.reset()
        q0 = self.qc.qubits.get_quantum_register_containing("q0")
        q1 = self.qc.qubits.get_quantum_register_containing("q1")
        q2 = self.qc.qubits.get_quantum_register_containing("q2")
        q3 = self.qc.qubits.get_quantum_register_containing("q3")
        q4 = self.qc.qubits.get_quantum_register_containing("q4")
        q0.set_state(State.state_from_string("10000"))
        self.qc.qubits.entangle_quantum_registers(q0, q1)
        self.qc.qubits.entangle_quantum_registers(q0, q2)
        self.qc.qubits.entangle_quantum_registers(q0, q3)
        self.qc.qubits.entangle_quantum_registers(q0, q4)

        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 10000 After: 11000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11000')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 11000 After: 01000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('01000')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 01000 After: 01000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('01000')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 01000 After: 01000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('01000')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 01000 After: 01100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('01100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 01100 After: 00100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 00100 After: 00100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00100')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q0")  # Before: 00100 After: 00100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00100')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q2")  # Before: 00100 After: 00100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 00100 After: 00100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('00100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 00100 After: 01100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('01100')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 01100 After: 11100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q2")  # Before: 11100 After: 11000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11000')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q1")  # Before: 11000 After: 11000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11000')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q0")  # Before: 11000 After: 11000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11000')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q3")  # Before: 11000 After: 11010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11010')))
        self.qc.apply_two_qubit_gate_CNOT("q3", "q2")  # Before: 11010 After: 11110
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11110')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q3")  # Before: 11110 After: 11100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11100')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 11100 After: 10100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10100')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q3")  # Before: 10100 After: 10100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10100')))
        self.qc.apply_two_qubit_gate_CNOT("q3", "q1")  # Before: 10100 After: 10100
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10100')))

        self.qc.apply_two_qubit_gate_CNOT("q0", "q4")  # Before: 10100 After: 10101
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10101')))
        self.qc.apply_two_qubit_gate_CNOT("q4", "q2")  # Before: 10101 After: 10001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10001')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q4")  # Before: 10001 After: 10001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10001')))
        self.qc.apply_two_qubit_gate_CNOT("q0", "q1")  # Before: 10001 After: 11001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11001')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q4")  # Before: 11001 After: 11000
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11000')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q4")  # Before: 11000 After: 11001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('11001')))
        self.qc.apply_two_qubit_gate_CNOT("q4", "q1")  # Before: 11001 After: 10001
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10001')))
        self.qc.apply_two_qubit_gate_CNOT("q4", "q3")  # Before: 10001 After: 10011
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10011')))
        self.qc.apply_two_qubit_gate_CNOT("q3", "q4")  # Before: 10011 After: 10010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10010')))
        self.qc.apply_two_qubit_gate_CNOT("q2", "q4")  # Before: 10010 After: 10010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10010')))
        self.qc.apply_two_qubit_gate_CNOT("q1", "q4")  # Before: 10010 After: 10010
        self.assertTrue(self.qc.qubit_states_equal("q0, q1, q2, q3, q4", State.state_from_string('10010')))

    def test_execute_bluestate(self):
        """Tests h, t, s, and bloch syntax on one qubit"""
        # This is a program to generate the 'blue state' in IBM's exercise
        self.qc.execute(Programs.program_blue_state.code)
        # check if we are in the blue state
        blue_state = Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.S * Gate.T * Gate.H * Gate.T * Gate.H * Gate.T * Gate.H * State.zero_state
        self.assertTrue(self.qc.bloch_coords_equal("q1", State.get_bloch(blue_state)))
        # check to make sure we didn't change any other qubits in the QC

        for unchanged_state in ["q0", "q2", "q3", "q4"]:
            self.assertTrue(self.qc.qubit_states_equal(unchanged_state, State.zero_state))

    def test_execute_X_Y_Z_Measure_Id_Sdag_Tdag(self):
        """Tests z, y, measure, id, sdag, tdag syntax on all 5 qubits"""
        self.qc.execute(Programs.program_test_XYZMeasureIdSdagTdag.code)
        # result should be 01101
        self.assertTrue(self.qc.qubit_states_equal("q0", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q2", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q3", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q4", State.one_state))

    def test_execute_cnot(self):
        """Tests cnot"""
        self.qc.execute(Programs.program_test_cnot.code)
        # result should be 01100
        self.assertTrue(self.qc.qubit_states_equal("q0", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q2", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q3", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q4", State.zero_state))

    def test_execute_many(self):
        """Tests z, y, cnot, measure, id, sdag, tdag syntax on all 5 qubits"""
        self.qc.execute(Programs.program_test_many.code)
        # result should be 01001
        self.assertTrue(self.qc.qubit_states_equal("q0", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q1", State.one_state))
        self.assertTrue(self.qc.qubit_states_equal("q2", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q3", State.zero_state))
        self.assertTrue(self.qc.qubit_states_equal("q4", State.one_state))

    # These tests will be enabled after entanglement is supported properly
    # # Bell state experiments
    def test_bellstate_programs(self):
        # This tests two qubit entanglement.
        for program, result_probs, result_cor in \
                zip(
                    [
                        Programs.program_zz,
                        Programs.program_zw,
                        Programs.program_zv,
                        Programs.program_xw,
                        Programs.program_xv
                    ],
                    [
                        (0.5, 0, 0, 0.5),
                        (0.426777, 0.073223, 0.073223, 0.426777),
                        (0.426777, 0.073223, 0.073223, 0.426777),
                        (0.426777, 0.073223, 0.073223, 0.426777),
                        (0.073223, 0.426777, 0.426777, 0.073223)
                    ],
                    [
                        1,
                        1 / sqrt(2),
                        1 / sqrt(2),
                        1 / sqrt(2),
                        -1 / sqrt(2)
                    ]
                ):
            self.qc.reset()
            self.qc.execute(program.code)
            state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
            probs = Probability.get_probabilities(state_before_measure)
            corex = Probability.get_correlated_expectation(state_before_measure)
            self.assertTrue(np.allclose(probs, result_probs))
            self.assertAlmostEqual(corex, result_cor)

    def test_ghz(self):
        program = Programs.program_ghz
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_ghz_measure_xxx(self):
        program = Programs.program_ghz_measure_xxx
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_ghz_measure_yyx(self):
        program = Programs.program_ghz_measure_yyx
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_ghz_measure_yxy(self):
        program = Programs.program_ghz_measure_yxy
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q0").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_ghz_measure_xyy(self):
        program = Programs.program_ghz_measure_xyy
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q0").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_swap_q0_q1(self):
        program = Programs.program_swap_q0_q1
        self.qc.reset()
        self.qc.execute(program.code)
        for qubit, bloch in zip(["q0", "q1", "q2", "q3", "q4"], program.bloch_vals):
            if bloch:
                self.assertTrue(self.qc.bloch_coords_equal(qubit, bloch))

    def test_program_controlled_hadamard(self):
        program = Programs.program_controlled_hadamard
        self.qc.reset()
        self.qc.execute(program.code)
        for qubit, bloch in zip(["q0", "q1", "q2", "q3", "q4"], program.bloch_vals):
            if bloch:
                self.assertTrue(self.qc.bloch_coords_equal(qubit, bloch))

    def test_reverse_cnot(self):
        program = Programs.program_reverse_cnot
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q2").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_swap(self):
        program = Programs.program_swap
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q2").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_approximate_sqrtT(self):
        program = Programs.program_approximate_sqrtT
        self.qc.reset()
        self.qc.execute(program.code)
        for qubit, bloch in zip(["q0", "q1", "q2", "q3", "q4"], program.bloch_vals):
            if bloch:
                self.assertTrue(self.qc.bloch_coords_equal(qubit, bloch))

    def test_program_toffoli_state_with_flips(self):
        program = Programs.program_toffoli_with_flips
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_toffoli_state(self):
        program = Programs.program_toffoli_state
        self.qc.reset()
        self.qc.execute(program.code)
        # we are going to reset things back to before they were measured
        on_qubit = self.qc.qubits.get_quantum_register_containing("q0")
        on_qubit.set_state(on_qubit.get_noop())
        self.assertTrue(self.qc.probabilities_equal("q0, q1, q2", np.array(program.result_probability)))

    def test_grover(self):
        for program in Programs.all_grover_tests:
            self.qc.reset()
            self.qc.execute(program.code)
            state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
            probs = Probability.get_probabilities(state_before_measure)
            self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_encoder_into_bitflip_code(self):
        # Simply fails
        program = Programs.program_encoder_into_bitflip_code
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability, atol=1e-3))

    def test_program_encoder_into_bitflip_code_parity_checks(self):
        program = Programs.program_encoder_into_bitflip_code_parity_checks
        self.qc.reset()
        self.qc.execute(program.code)
        # we are going to reset things back to before they were measured
        on_qubit = self.qc.qubits.get_quantum_register_containing("q0")
        on_qubit.set_state(on_qubit.get_noop())
        self.assertTrue(self.qc.probabilities_equal("q0, q1, q2, q3, q4", np.array(program.result_probability)))

    def test_program_deutschjozsa_constant_n3(self):
        program = Programs.program_deutschjozsa_n3
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_deutschjozsa_n3(self):
        program = Programs.program_deutschjozsa_n3
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q1").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_plaquette_code(self):
        for program in Programs.all_normal_plaquette_programs:
            self.qc.reset()
            self.qc.execute(program.code)
            state_before_measure = self.qc.qubits.get_quantum_register_containing("q2").get_noop()
            probs = Probability.get_probabilities(state_before_measure)
            self.assertTrue(np.allclose(probs, program.result_probability))

    def test_plaquette_zXplusminusplusminus(self):
        program = Programs.program_plaquette_zXplusminusplusminus
        self.qc.reset()
        self.qc.execute(program.code)
        state_before_measure = self.qc.qubits.get_quantum_register_containing("q2").get_noop()
        probs = Probability.get_probabilities(state_before_measure)
        self.assertTrue(np.allclose(probs, program.result_probability))

    def test_program_encoder_and_decoder_tomography(self):
        program = Programs.program_encoder_and_decoder_tomography
        self.qc.reset()
        self.qc.execute(program.code)
        for qubit_name, bloch in zip(["q0", "q1", "q2", "q3", "q4"], program.bloch_vals):
            if bloch:
                self.assertTrue(self.qc.bloch_coords_equal(qubit_name, bloch))

if __name__ == '__main__':
    unittest.main()
