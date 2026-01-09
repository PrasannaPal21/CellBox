"""
Tests for dataset module in cellbox.dataset.
"""
import pytest
import numpy as np
import pandas as pd
import os


class TestDataFilesExist:
    """Test that sample data files exist in the project."""
    
    def test_pert_csv_exists(self, project_root):
        """pert.csv should exist in data folder."""
        path = os.path.join(project_root, "data", "pert.csv")
        assert os.path.exists(path), f"Missing {path}"
    
    def test_expr_csv_exists(self, project_root):
        """expr.csv should exist in data folder."""
        path = os.path.join(project_root, "data", "expr.csv")
        assert os.path.exists(path), f"Missing {path}"
    
    def test_node_index_exists(self, project_root):
        """node_Index.csv should exist in data folder."""
        path = os.path.join(project_root, "data", "node_Index.csv")
        assert os.path.exists(path), f"Missing {path}"
    
    def test_loo_label_exists(self, project_root):
        """loo_label.csv should exist in data folder."""
        path = os.path.join(project_root, "data", "loo_label.csv")
        assert os.path.exists(path), f"Missing {path}"


class TestDataFileFormats:
    """Test that data files have expected format."""
    
    def test_pert_csv_format(self, project_root):
        """pert.csv should have correct dimensions."""
        path = os.path.join(project_root, "data", "pert.csv")
        df = pd.read_csv(path, header=None)
        
        # should have rows and columns
        assert df.shape[0] > 0, "pert.csv should have rows"
        assert df.shape[1] > 0, "pert.csv should have columns"
    
    def test_expr_csv_format(self, project_root):
        """expr.csv should have correct dimensions."""
        path = os.path.join(project_root, "data", "expr.csv")
        df = pd.read_csv(path, header=None)
        
        assert df.shape[0] > 0, "expr.csv should have rows"
        assert df.shape[1] > 0, "expr.csv should have columns"
    
    def test_pert_expr_same_rows(self, project_root):
        """pert and expr should have same number of samples."""
        pert_path = os.path.join(project_root, "data", "pert.csv")
        expr_path = os.path.join(project_root, "data", "expr.csv")
        
        pert_df = pd.read_csv(pert_path, header=None)
        expr_df = pd.read_csv(expr_path, header=None)
        
        assert pert_df.shape[0] == expr_df.shape[0], \
            "pert and expr should have same number of samples"
    
    def test_node_index_format(self, project_root):
        """node_Index.csv should have node names."""
        path = os.path.join(project_root, "data", "node_Index.csv")
        df = pd.read_csv(path, header=None)
        
        assert df.shape[0] > 0, "node_Index should have entries"


class TestNpzDataFiles:
    """Test npz data files if they exist."""
    
    def test_pert_subset_npz(self, project_root):
        """pert_subset.npz should be loadable."""
        path = os.path.join(project_root, "data", "pert_subset.npz")
        if os.path.exists(path):
            from scipy import sparse
            data = sparse.load_npz(path)
            assert data.shape[0] > 0
    
    def test_expr_subset_npz(self, project_root):
        """expr_subset.npz should be loadable."""
        path = os.path.join(project_root, "data", "expr_subset.npz")
        if os.path.exists(path):
            from scipy import sparse
            data = sparse.load_npz(path)
            assert data.shape[0] > 0
