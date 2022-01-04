import unittest

import numpy as np

from environment import Environment, bin2dec

class EnvironmentSetUp(unittest.TestCase):
    """Defines Random MDP Environment"""

    def setUp(self):
        """Code here will run before every test"""

        self.n_states = 10
        self.n_actions = 2
        self.n_nodes = 3
        self.n_phi = 5
        self.n_varphi = 7
        self.seed=42

        self.env = Environment(
            n_states=self.n_states,
            n_actions=self.n_actions,
            n_nodes=self.n_nodes,
            n_phi=self.n_phi,
            n_varphi=self.n_varphi,
            seed=self.seed
        )


class TestEnvironmentProperties(EnvironmentSetUp):
    """Tests

        * Input parameters.
        * Auxiliary variables.
        * Auxiliary functions.
        * Markov decision process variables.

    """
    
    def setUp(self):
        super(TestEnvironmentProperties, self).setUp()

        n_action_space = self.n_actions ** self.n_nodes
        self.P_shape = (self.n_states * n_action_space, self.n_states)
        self.PHI_shape = (self.n_states * n_action_space, self.n_phi) 
        self.VARPHI_shape = (self.n_states, self.n_actions, self.n_nodes, self.n_varphi) 
        self.R_shape = (self.n_states * n_action_space, self.n_nodes) 

    # Test input parameters.
    def test_n_states(self):
        # State.
        self.assertEqual(self.env.n_states, self.n_states)

    def test_n_actions(self):
        # Actions
        self.assertEqual(self.env.n_actions, self.n_actions)

    def test_n_nodes(self):
        # Nodes
        self.assertEqual(self.env.n_nodes, self.n_nodes)

    def test_n_phi(self):
        # Phi
        self.assertEqual(self.env.n_phi, self.n_phi)

    def test_n_varphi(self):
        # Varphi
        self.assertEqual(self.env.n_varphi, self.n_varphi)

    def test_n_action_space(self):
        # Action space
        self.assertEqual(self.env.n_action_space, self.n_actions ** self.n_nodes)
        
    # Test auxiliary variables.
    def test_n_edges(self):
        # connectivity_ratio = 2 * n_edges / (n_nodes)*(n_nodes - 1) 
        # requirement:4 / n_nodes <--> n_edges = 2 * (n_nodes - 1)
        self.assertEqual(self.env.n_edges, 2 * (self.n_nodes - 1))

    def test_seed(self):
        # seed
        self.assertEqual(self.env.seed, self.seed)

    def test_n_action_space(self):
        # n_action_space
        self.assertEqual(self.env.n_action_space, self.n_actions ** self.n_nodes)

    # Test MDP variables.
    def test_P_shape(self):
        self.assertEqual(self.env.P.shape, self.P_shape)

    def test_P_stochastic(self):
        # all columns must sum to one.
        test = self.env.P @ np.ones(self.P_shape[1])
        ret = np.ones(self.P_shape[0]) 
        np.testing.assert_almost_equal(test, ret)

    def test_P_ergodic(self):
        # there must not be a zero-value.
        thresh = (1/ self.n_states) * 10e-5  
        self.assertFalse(np.any(self.env.P < thresh))

    def test_PHI_shape(self):
        # critic features.
        self.assertEqual(self.env.PHI.shape, self.PHI_shape)

    def test_VARPHI_shape(self):
        # actor features.
        self.assertEqual(self.env.VARPHI.shape, self.VARPHI_shape)

    def test_R_shape(self):
        # average rewards
        self.assertEqual(self.env.R.shape, self.R_shape)

    def test_R_bounds(self):
        # average rewards must be greater than 0
        # average rewards must be lesser than 4
        self.assertFalse(np.any(self.env.R < 0) or np.any(self.env.R > 3.99999))

    def test_action_000(self):
        self.assertEqual(bin2dec([0, 0, 0]), 0)

    def test_action_100(self):
        self.assertEqual(bin2dec([1, 0, 0]), 1)

    def test_action_010(self):
        self.assertEqual(bin2dec([0, 1, 0]), 2)

    def test_action_110(self):
        self.assertEqual(bin2dec([1, 1, 0]), 3)

    def test_action_001(self):
        self.assertEqual(bin2dec([0, 0, 1]), 4)

    def test_action_101(self):
        self.assertEqual(bin2dec([1, 0, 1]), 5)

    def test_action_011(self):
        self.assertEqual(bin2dec([0, 1, 1]), 6)

    def test_action_111(self):
        self.assertEqual(bin2dec([1, 1, 1]), 7)

    def test_adjacency(self):
        res = np.ones((self.n_nodes, self.n_nodes))
        np.testing.assert_almost_equal(self.env.adjacency, res)

    def tearDown(self):
        pass

class TestEnvironmentGetters(EnvironmentSetUp):
    """Tests Getters

        * get_features
        * get_phi
        * get_varphi
        * get_reward

    """
    def setUp(self):
        super(TestEnvironmentGetters, self).setUp()
        self.current_state = self.env.state

    def test_get_dim(self):
        states = [s for s in range(self.n_states) if s != self.current_state]
        arbitrary_state = np.random.choice(states)
        arbitrary_action = np.array([1, 1, 0])

        state_index = arbitrary_state * self.n_actions ** self.n_nodes
        action_index = bin2dec(arbitrary_action) 
        dim = state_index + action_index 
        test = self.env.get_dim(arbitrary_state, arbitrary_action)
        self.assertEqual(dim, test)

    def test_get_varphi_current(self):
        varphi = self.env.VARPHI[self.current_state, ...]
        np.testing.assert_almost_equal(varphi, self.env.get_varphi())

    def test_get_varphi_arbitrary(self):
        states = [s for s in range(self.n_states) if s != self.current_state]
        arbitrary_state = np.random.choice(states)
        varphi = self.env.VARPHI[arbitrary_state, ...]
        np.testing.assert_almost_equal(varphi, self.env.get_varphi(arbitrary_state))

    def test_get_phi_arbitrary_action(self):
        arbitrary_action = np.array([1, 0, 1])
        state_index = self.current_state * self.n_actions ** self.n_nodes
        action_index = bin2dec(arbitrary_action) 
        phi = self.env.PHI[state_index + action_index, ...]
        np.testing.assert_almost_equal(phi, self.env.get_phi(arbitrary_action)) 

    def test_get_phi_arbitrary_state_and_action(self):
        states = [s for s in range(self.n_states) if s != self.current_state]
        arbitrary_state = np.random.choice(states)
        arbitrary_action = np.array([0, 1, 1])

        dim = arbitrary_state * self.n_actions ** self.n_nodes + \
                bin2dec(arbitrary_action) 

        phi = self.env.PHI[dim, ...]
        test = self.env.get_phi(arbitrary_action, arbitrary_state)
        np.testing.assert_almost_equal(phi, test) 

    def test_get_features_no_action(self):
        np.testing.assert_almost_equal(self.env.get_features(), self.env.get_varphi())

    def test_get_features_arbitrary_action(self):
        arbitrary_action = np.array([0, 1, 1])
        test = self.env.get_features(arbitrary_action)

        np.testing.assert_almost_equal(test[0], self.env.get_phi(arbitrary_action))
        np.testing.assert_almost_equal(test[1], self.env.get_varphi())

    def test_get_consensus(self):
        test = self.env.get_consensus()
        # metropolis weights
        res = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
        np.testing.assert_almost_equal(test, res)


class TestEnvironmentRoutines(EnvironmentSetUp):
    """Tests Routines"""

    def setUp(self):
        super(TestEnvironmentRoutines, self).setUp()
        self.initial_state = self.env.state
        self.arbitrary_action = np.array([0, 0, 1])
        self.env.next_step(self.arbitrary_action)
        self.next_state = self.env.state 
        self.env.reset()


    def test_reset(self):
        self.assertEqual(self.env.state, self.initial_state)
        self.assertEqual(self.env.n_step, 0)

    def test_next_step(self):
        self.env.next_step(self.arbitrary_action)

        self.assertEqual(self.env.state, self.next_state)
        self.assertEqual(self.env.n_step, 1)

    def test_loop(self):
        gen = self.env.loop(2)
        test = next(gen)
        np.testing.assert_almost_equal(self.env.get_varphi(), test)
        self.assertEqual(self.env.n_step, 0)

        next_features, reward, done = gen.send(self.arbitrary_action)
        
        np.testing.assert_almost_equal(next_features[0], self.env.get_phi(self.arbitrary_action))
        np.testing.assert_almost_equal(next_features[1], self.env.get_varphi())
        
        self.assertTrue(done)
        self.assertEqual(self.env.n_step, 1)
        self.assertEqual(self.env.state, self.next_state)

        # assertRaises receives a callable
        with self.assertRaises(StopIteration):
            gen.send(self.arbitrary_action)

if __name__ == '__main__':
    unittest.main()
