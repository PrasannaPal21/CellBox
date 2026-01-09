"""
Tests for utility functions in cellbox.utils module.
"""
import pytest
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TestLossFunction:
    """Test the loss calculation function."""
    
    def test_mse_loss_zero_when_identical(self):
        """MSE should be zero when prediction matches ground truth."""
        from cellbox.utils import loss
        
        tf.reset_default_graph()
        
        x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        x_hat = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        W = tf.Variable(tf.zeros((3, 3)))
        
        total_loss, mse = loss(x, x_hat, W)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mse_val = sess.run(mse)
        
        assert mse_val == pytest.approx(0.0)
    
    def test_mse_loss_calculation(self):
        """MSE should be mean of squared differences."""
        from cellbox.utils import loss
        
        tf.reset_default_graph()
        
        # x_gold = [1, 2], x_hat = [2, 4]
        # diff = [1, 2], squared = [1, 4], mean = 2.5
        x = tf.constant([[1.0, 2.0]], dtype=tf.float32)
        x_hat = tf.constant([[2.0, 4.0]], dtype=tf.float32)
        W = tf.Variable(tf.zeros((2, 2)))
        
        total_loss, mse = loss(x, x_hat, W)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mse_val = sess.run(mse)
        
        assert mse_val == pytest.approx(2.5)
    
    def test_l1_regularization(self):
        """L1 regularization should add sum of absolute weights."""
        from cellbox.utils import loss
        
        tf.reset_default_graph()
        
        x = tf.constant([[0.0]], dtype=tf.float32)
        x_hat = tf.constant([[0.0]], dtype=tf.float32)
        W = tf.Variable([[1.0, -2.0], [3.0, -4.0]])  # sum of abs = 10
        
        total_loss, mse = loss(x, x_hat, W, l1=0.1)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_val = sess.run(total_loss)
        
        # mse=0, l1=0.1*10=1.0
        assert total_val == pytest.approx(1.0)
    
    def test_l2_regularization(self):
        """L2 regularization should add sum of squared weights."""
        from cellbox.utils import loss
        
        tf.reset_default_graph()
        
        x = tf.constant([[0.0]], dtype=tf.float32)
        x_hat = tf.constant([[0.0]], dtype=tf.float32)
        W = tf.Variable([[1.0, 2.0]])  # sum of squares = 1+4 = 5
        
        total_loss, mse = loss(x, x_hat, W, l2=0.1)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_val = sess.run(total_loss)
        
        # mse=0, l2=0.1*5=0.5
        assert total_val == pytest.approx(0.5)


class TestMD5:
    """Test MD5 hash utility."""
    
    def test_md5_returns_string(self):
        """md5 should return a hex string."""
        from cellbox.utils import md5
        
        class Obj:
            def __init__(self):
                self.a = 1
                self.b = "test"
        
        result = md5(Obj())
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex length
    
    def test_md5_deterministic(self):
        """Same object should produce same hash."""
        from cellbox.utils import md5
        
        class Obj:
            def __init__(self):
                self.value = 42
        
        obj1 = Obj()
        obj2 = Obj()
        
        assert md5(obj1) == md5(obj2)
    
    def test_md5_different_for_different_values(self):
        """Different objects should produce different hashes."""
        from cellbox.utils import md5
        
        class Obj1:
            def __init__(self):
                self.x = 1
        
        class Obj2:
            def __init__(self):
                self.x = 2
        
        assert md5(Obj1()) != md5(Obj2())


class TestTimeLogger:
    """Test TimeLogger utility class."""
    
    def test_logger_creation(self):
        """Should create logger with default values."""
        from cellbox.utils import TimeLogger
        
        logger = TimeLogger()
        assert logger.time_logger_step == 1
        assert logger.hierachy == 1
        assert logger.step_count == 0
    
    def test_logger_custom_params(self):
        """Should accept custom parameters."""
        from cellbox.utils import TimeLogger
        
        logger = TimeLogger(time_logger_step=5, hierachy=2)
        assert logger.time_logger_step == 5
        assert logger.hierachy == 2


class TestOptimize:
    """Test the optimize function."""
    
    def test_optimize_returns_op(self):
        """optimize should return a tensorflow operation."""
        from cellbox.utils import optimize
        
        tf.reset_default_graph()
        
        x = tf.Variable(1.0)
        loss_tensor = x ** 2
        lr = tf.placeholder(tf.float32)
        
        opt_op = optimize(loss_tensor, lr)
        
        # should be able to run without error
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(opt_op, feed_dict={lr: 0.01})
            new_x = sess.run(x)
        
        # x should have decreased (gradient descent on x^2)
        assert new_x < 1.0
