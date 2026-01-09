"""
Tests for ODE kernel functions in cellbox.kernel module.
"""
import pytest
import numpy as np


# need to import tf and set up before importing cellbox
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TestEnvelopeFunctions:
    """Test envelope function factory."""
    
    def test_tanh_envelope(self):
        """tanh envelope should use tf.tanh."""
        from cellbox.kernel import get_envelope
        
        class MockArgs:
            envelope_form = "tanh"
        
        args = MockArgs()
        envelope_fn = get_envelope(args)
        
        # test with simple tensor
        with tf.Session() as sess:
            x = tf.constant([0.0, 1.0, -1.0])
            result = sess.run(envelope_fn(x))
        
        expected = np.tanh([0.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_polynomial_envelope_odd_k(self):
        """Polynomial envelope with odd k."""
        from cellbox.kernel import get_envelope
        
        class MockArgs:
            envelope_form = "polynomial"
            polynomial_k = 3
        
        args = MockArgs()
        envelope_fn = get_envelope(args)
        
        with tf.Session() as sess:
            x = tf.constant([0.0, 1.0, -1.0])
            result = sess.run(envelope_fn(x))
        
        # x^3 / (1 + |x|^3) for each value
        # 0^3/(1+0) = 0, 1^3/(1+1) = 0.5, (-1)^3/(1+1) = -0.5
        expected = [0.0, 0.5, -0.5]
        np.testing.assert_array_almost_equal(result, expected)


class TestODESolvers:
    """Test ODE solver implementations."""
    
    def test_euler_solver_shape(self):
        """Euler solver should return correct shape."""
        from cellbox.kernel import euler_solver
        
        n_x = 5
        batch_size = 3
        n_T = 10
        dT = 0.1
        
        x = tf.zeros((n_x, batch_size))
        t_mu = tf.zeros((n_x, batch_size))
        dxdt_fn = lambda x, mu: x  # simple identity
        
        result = euler_solver(x, t_mu, dT, n_T, dxdt_fn)
        
        with tf.Session() as sess:
            out = sess.run(result)
        
        # should be (n_T, n_x, batch_size)
        assert out.shape == (n_T, n_x, batch_size)
    
    def test_heun_solver_shape(self):
        """Heun solver should return correct shape."""
        from cellbox.kernel import heun_solver
        
        n_x = 5
        batch_size = 3
        n_T = 10
        dT = 0.1
        
        x = tf.zeros((n_x, batch_size))
        t_mu = tf.zeros((n_x, batch_size))
        dxdt_fn = lambda x, mu: x
        
        result = heun_solver(x, t_mu, dT, n_T, dxdt_fn)
        
        with tf.Session() as sess:
            out = sess.run(result)
        
        assert out.shape == (n_T, n_x, batch_size)
    
    def test_rk4_solver_shape(self):
        """RK4 solver should return correct shape."""
        from cellbox.kernel import rk4_solver
        
        n_x = 4
        batch_size = 2
        n_T = 5
        dT = 0.1
        
        x = tf.zeros((n_x, batch_size))
        t_mu = tf.zeros((n_x, batch_size))
        dxdt_fn = lambda x, mu: -x  # decay
        
        result = rk4_solver(x, t_mu, dT, n_T, dxdt_fn)
        
        with tf.Session() as sess:
            out = sess.run(result)
        
        assert out.shape == (n_T, n_x, batch_size)

    def test_midpoint_solver_shape(self):
        """Midpoint solver should return correct shape."""
        from cellbox.kernel import midpoint_solver
        
        n_x = 3
        batch_size = 4
        n_T = 8
        dT = 0.1
        
        x = tf.zeros((n_x, batch_size))
        t_mu = tf.zeros((n_x, batch_size))
        dxdt_fn = lambda x, mu: x * 0.5
        
        result = midpoint_solver(x, t_mu, dT, n_T, dxdt_fn)
        
        with tf.Session() as sess:
            out = sess.run(result)
        
        assert out.shape == (n_T, n_x, batch_size)


class TestGetODESolver:
    """Test ODE solver factory function."""
    
    def test_get_heun_solver(self):
        """Should return heun solver."""
        from cellbox.kernel import get_ode_solver, heun_solver
        
        class MockArgs:
            ode_solver = "heun"
        
        solver = get_ode_solver(MockArgs())
        assert solver == heun_solver
    
    def test_get_euler_solver(self):
        """Should return euler solver."""
        from cellbox.kernel import get_ode_solver, euler_solver
        
        class MockArgs:
            ode_solver = "euler"
        
        solver = get_ode_solver(MockArgs())
        assert solver == euler_solver
    
    def test_get_rk4_solver(self):
        """Should return rk4 solver."""
        from cellbox.kernel import get_ode_solver, rk4_solver
        
        class MockArgs:
            ode_solver = "rk4"
        
        solver = get_ode_solver(MockArgs())
        assert solver == rk4_solver

    def test_invalid_solver_raises(self):
        """Should raise for invalid solver name."""
        from cellbox.kernel import get_ode_solver
        
        class MockArgs:
            ode_solver = "invalid_solver"
        
        with pytest.raises(Exception):
            get_ode_solver(MockArgs())
