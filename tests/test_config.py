"""
Tests for the Config class in cellbox.config module.
"""
import pytest
import os
import json
import tempfile


class TestConfigDefaults:
    """Test that Config sets proper default values."""
    
    def test_experiment_id_default(self, temp_config_file):
        """Should default to 'Debugging' when not specified."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert cfg.experiment_id == "Debugging"
    
    def test_model_prefix_default(self, temp_config_file):
        """Default model prefix should be 'seed'."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert cfg.model_prefix == "seed"

    def test_ckpt_name_default(self, temp_config_file):
        """Checkpoint name should default to model.ckpt."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert cfg.ckpt_name == "model.ckpt"

    def test_batchsize_default(self, temp_config_file):
        """Batch size should default to 8."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert cfg.batchsize == 8


class TestConfigParsing:
    """Test that Config correctly parses values from JSON."""
    
    def test_parses_n_x(self, temp_config_file):
        """Should parse n_x from config file."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert cfg.n_x == 10
    
    def test_parses_stages(self, temp_config_file):
        """Should parse stages list correctly."""
        from cellbox.config import Config
        cfg = Config(temp_config_file)
        assert len(cfg.stages) == 1
        assert cfg.stages[0]["nT"] == 5

    def test_custom_experiment_id(self):
        """Should use custom experiment_id when provided."""
        from cellbox.config import Config
        
        config_data = {
            "experiment_id": "my_experiment",
            "n_x": 10,
            "stages": []
        }
        fd, path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(config_data, f)
            cfg = Config(path)
            assert cfg.experiment_id == "my_experiment"
        finally:
            os.unlink(path)


class TestConfigValidation:
    """Test config validation for supported options."""
    
    def test_valid_ode_solver(self):
        """Should accept valid ODE solver options."""
        from cellbox.config import Config
        
        for solver in ["heun", "euler", "rk4", "midpoint"]:
            config_data = {"n_x": 10, "ode_solver": solver, "stages": []}
            fd, path = tempfile.mkstemp(suffix='.json')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(config_data, f)
                cfg = Config(path)
                assert cfg.ode_solver == solver
            finally:
                os.unlink(path)
    
    def test_valid_envelope_form(self):
        """Should accept tanh, polynomial, and hill envelope forms."""
        from cellbox.config import Config
        
        for form in ["tanh", "polynomial", "hill"]:
            config_data = {"n_x": 10, "envelope_form": form, "stages": []}
            fd, path = tempfile.mkstemp(suffix='.json')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(config_data, f)
                cfg = Config(path)
                assert cfg.envelope_form == form
            finally:
                os.unlink(path)


class TestConfigFileHandling:
    """Test file I/O related functionality."""
    
    def test_missing_file_raises_error(self):
        """Should raise FileNotFoundError for missing config file."""
        from cellbox.config import Config
        
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.json")
    
    def test_invalid_json_raises_error(self):
        """Should raise error for malformed JSON."""
        from cellbox.config import Config
        
        fd, path = tempfile.mkstemp(suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write("{ invalid json }")
            with pytest.raises(json.JSONDecodeError):
                Config(path)
        finally:
            os.unlink(path)
