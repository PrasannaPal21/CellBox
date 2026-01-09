# Tests

Unit tests for the CellBox package.

## Running the tests

Make sure you have pytest installed:

```bash
pip install pytest
```

Then from the project root, run:

```bash
pytest tests/
```

Or if you want more details:

```bash
pytest tests/ -v
```

## What's being tested

### test_config.py
Tests for the Config class - checking that it parses JSON correctly, sets proper defaults, and validates options like `ode_solver` and `envelope_form`.

### test_kernel.py
Tests for ODE solvers (euler, heun, rk4, midpoint) and envelope functions. Mostly checking that the output shapes are correct and the factory functions return the right solvers.

### test_utils.py
Tests for utility functions - loss calculation with L1/L2 regularization, the MD5 hash function, TimeLogger class, and the optimizer.

### test_dataset.py
Validates that the data files exist and have the expected format. Also checks that pert.csv and expr.csv have matching row counts.

## Adding new tests

Just add a new `test_*.py` file in this folder. Pytest will pick it up automatically.

Fixtures are in `conftest.py` - you can use `project_root`, `sample_config_dict`, and `temp_config_file` in your tests.

## Notes

- The tests use TensorFlow v1 compat mode (same as the main package)
- Some deprecation warnings are filtered out in pytest.ini to keep the output clean
- Tests are designed to not modify any existing files or state
