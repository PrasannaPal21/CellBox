"""
Shared fixtures for CellBox tests.
"""
import pytest
import os
import tempfile
import json


@pytest.fixture
def sample_config_dict():
    """Basic config dictionary for testing."""
    return {
        "model": "CellBox",
        "n_x": 10,
        "n_protein_nodes": 5,
        "n_activity_nodes": 8,
        "pert_file": "data/pert.csv",
        "expr_file": "data/expr.csv",
        "n_epoch": 2,
        "n_iter": 10,
        "stages": [{
            "nT": 5,
            "sub_stages": [{"lr_val": 0.1}]
        }]
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Creates a temporary config file for testing."""
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(sample_config_dict, f)
        yield path
    finally:
        os.unlink(path)


@pytest.fixture
def project_root():
    """Returns the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
