import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import json
import struct
import math

# Import the module to be tested
from di_lib.seismic_cube import (
    scalar_prod, subtract, add, mult_by_scalar, round,
    join_time_axes, recalculate_trace_to_new_time_axis,
    generate_fragments_grid_incr, generate_fragments_grid,
    DISeismicCube, DISeismicCubeWriter, MAXFLOAT
)

class TestUtilityFunctions(unittest.TestCase):
    """Tests for the utility functions in seismic_cube.py"""
    
    def test_scalar_prod(self):
        """Test scalar product function"""
        self.assertEqual(scalar_prod([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(scalar_prod([], []), 0)
        self.assertEqual(scalar_prod([0, 0, 0], [1, 2, 3]), 0)
        
    def test_subtract(self):
        """Test vector subtraction function"""
        self.assertEqual(subtract((1, 2, 3), (4, 5, 6)), (-3, -3, -3))
        self.assertEqual(subtract((10, 20), (5, 5)), (5, 15))
        self.assertEqual(subtract((), ()), ())
        
    def test_add(self):
        """Test vector addition function"""
        self.assertEqual(add((1, 2, 3), (4, 5, 6)), (5, 7, 9))
        self.assertEqual(add((10, 20), (5, 5)), (15, 25))
        self.assertEqual(add((), ()), ())
        
    def test_mult_by_scalar(self):
        """Test multiplication by scalar function"""
        self.assertEqual(mult_by_scalar((1, 2, 3), 2), (2, 4, 6))
        self.assertEqual(mult_by_scalar((10, 20), 0.5), (5, 10))
        self.assertEqual(mult_by_scalar((), 5), ())
        
    def test_round(self):
        """Test rounding function"""
        self.assertEqual(round(1.4), 1)
        self.assertEqual(round(1.5), 2)
        self.assertEqual(round(1.6), 2)
        self.assertEqual(round(-1.5), -1)
        
    def test_join_time_axes(self):
        """Test join_time_axes function"""
        # Test case where t1 has lower origin
        t1 = (100, 2, 50)  # origin=100, step=2, n=50
        t2 = (110, 4, 30)  # origin=110, step=4, n=30
        expected = (110, 2, 70)  # origin=110, step=2, n=70
        self.assertEqual(join_time_axes(t1, t2), expected)
        
        # Test case where t2 has lower origin
        t1 = (200, 4, 25)
        t2 = (150, 2, 40)
        expected = (150, 2, 125)
        self.assertEqual(join_time_axes(t1, t2), expected)
        
        # Test case with same origins but different steps
        t1 = (100, 4, 50)
        t2 = (100, 2, 80)
        expected = (100, 2, 159)
        self.assertEqual(join_time_axes(t1, t2), expected)
        
    def test_recalculate_trace_to_new_time_axis(self):
        """Test recalculate_trace_to_new_time_axis function"""
        trace = [1.0, 2.0, 3.0, 4.0, 5.0]
        t1 = (10, 2, 5)  # origin=10, step=2, n=5
        
        # Same time axis - should return same values
        t_new = (10, 2, 5)
        result = recalculate_trace_to_new_time_axis(trace, t1, t_new)
        self.assertEqual(result, (1.0, 2.0, 3.0, 4.0, 5.0))
        
        # New time axis with finer step
        t_new = (10, 1, 9)
        result = recalculate_trace_to_new_time_axis(trace, t1, t_new)
        expected = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)
            
        # New time axis outside original range
        t_new = (8, 2, 3)
        result = recalculate_trace_to_new_time_axis(trace, t1, t_new)
        self.assertTrue(result[0] > MAXFLOAT * 0.9)  # Should be MAXFLOAT
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], 2.0)
        
    def test_generate_fragments_grid_incr(self):
        """Test generate_fragments_grid_incr function"""
        result = generate_fragments_grid_incr(1, 11, 5, 1, 6, 2)
        expected = [
            (1, 1, 5, 2), (1, 3, 5, 4), (1, 5, 5, 5),
            (6, 1, 10, 2), (6, 3, 10, 4), (6, 5, 10, 5)
        ]
        self.assertEqual(result, expected)
        
    def test_generate_fragments_grid(self):
        """Test generate_fragments_grid function"""
        # Test with default min values
        result = generate_fragments_grid(None, 11, 2, None, 6, 2)
        expected = [(1, 1, 5, 3), (1, 4, 5, 5), (6, 1, 10, 3), (6, 4, 10, 5)]
        self.assertEqual(result, expected)
        
        # Test with custom min values
        result = generate_fragments_grid(5, 15, 2, 10, 20, 2)
        expected = [(5, 10, 9, 14), (5, 15, 9, 19), (10, 10, 14, 14), (10, 15, 14, 19)]
        self.assertEqual(result, expected)


class TestDISeismicCube(unittest.TestCase):
    """Tests for the DISeismicCube class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cube = DISeismicCube(100, "test_geometry", "test_name", "test_name2")
        self.cube.server_url = "http://test.server"
        self.cube.token = "test_token"
        
        # Sample cube info for testing
        self.cube_info = {
            "id": 123,
            "geometry_id": 456,
            "geometry_name": "test_geometry",
            "name": "test_name",
            "name2": "test_name2",
            "origin": [100, 200, 300],
            "d_inline": [1, 0, 0],
            "d_xline": [0, 1, 0],
            "max_inline": 100,
            "max_xline": 200,
            "nz": 300,
            "z_step": 4,
            "domain": "time",
            "z_start": 0,
            "min_inline": 1,
            "min_xline": 1
        }

    @patch('requests.get')
    def test_read_info(self, mock_get):
        """Test _read_info method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps([self.cube_info]).encode()
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Execute
        self.cube._read_info()
        
        # Verify
        mock_get.assert_called_once_with(f"{self.cube.server_url}/seismic_3d/list/{self.cube.project_id}/")
        self.assertEqual(self.cube.cube_id, 123)
        self.assertEqual(self.cube.geometry_id, 456)
        self.assertEqual(self.cube.origin, [100, 200, 300])
        self.assertEqual(self.cube.v_i, [1, 0, 0])
        self.assertEqual(self.cube.v_x, [0, 1, 0])
        self.assertEqual(self.cube.n_i, 101)
        self.assertEqual(self.cube.n_x, 201)
        self.assertEqual(self.cube.n_samples, 300)
        self.assertEqual(self.cube.time_step, 4)
        self.assertEqual(self.cube.domain, "time")
        self.assertEqual(self.cube.data_start, 0)
        self.assertEqual(self.cube.min_i, 1)
        self.assertEqual(self.cube.min_x, 1)

    def test_get_info(self):
        """Test _get_info method"""
        # Setup cube with test values
        self.cube.cube_id = 123
        self.cube.geometry_id = 456
        self.cube.origin = [100, 200, 300]
        self.cube.v_i = [1, 0, 0]
        self.cube.v_x = [0, 1, 0]
        self.cube.n_i = 101
        self.cube.n_x = 201
        self.cube.n_samples = 300
        self.cube.time_step = 4
        self.cube.domain = "time"
        self.cube.data_start = 0
        self.cube.min_i = 1
        self.cube.min_x = 1
        
        # Execute
        info = self.cube._get_info()
        
        # Verify
        expected = {
            "geometry_name": "test_geometry",
            "geometry_id": 456,
            "name": "test_name",
            "name2": "test_name2",
            "max_inline": 100,
            "max_xline": 200,
            "nz": 300,
            "origin": [100, 200, 300],
            "d_inline": [1, 0, 0],
            "d_xline": [0, 1, 0],
            "domain": "time",
            "z_start": 0,
            "z_step": 4,
            "min_inline": 1,
            "min_xline": 1,
            "id": 123
        }
        self.assertEqual(info, expected)

    @patch('requests.get')
    def test_get_fragment(self, mock_get):
        """Test get_fragment method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Create sample data
        nz, ncdps, ninlines = 10, 5, 3
        data = np.random.random((ninlines, ncdps, nz)).astype(np.float32)
        header = struct.pack('<iii', nz, ncdps, ninlines)
        mock_response.content = header + data.tobytes()
        
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Execute
        result = self.cube.get_fragment(1, 3, 5, 5)
        
        # Verify
        mock_get.assert_called_once_with(
            f"{self.cube.server_url}/seismic_3d/data/rect_fragment/{self.cube.cube_id}/"
            f"?inline_no=1&inline_count=3&xline_no=5&xline_count=5"
        )
        self.assertEqual(result.shape, (ninlines, ncdps, nz))
        
    @patch('requests.get')
    def test_get_inline(self, mock_get):
        """Test get_inline method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Create sample data for trimmed=True
        nz, ninlines = 10, 5
        start_i = 3
        data = np.random.random((ninlines, nz)).astype(np.float32)
        header = struct.pack('<iii', start_i, nz, ninlines)
        mock_response.content = header + data.tobytes()
        
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Execute for trimmed=True
        result = self.cube.get_inline(15, trimmed=True)
        
        # Verify
        mock_get.assert_called_once_with(
            f"{self.cube.server_url}/seismic_3d/data/inline_trim/{self.cube.cube_id}/?inline_no=15"
        )
        self.assertEqual(result.shape, (start_i + ninlines, nz))
        
        # Reset mock
        mock_get.reset_mock()
        
        # Create sample data for trimmed=False
        data = np.random.random((ninlines, nz)).astype(np.float32)
        header = struct.pack('<ii', nz, ninlines)
        mock_response.content = header + data.tobytes()
        
        # Execute for trimmed=False
        result = self.cube.get_inline(15, trimmed=False)
        
        # Verify
        mock_get.assert_called_once_with(
            f"{self.cube.server_url}/seismic_3d/data/inline/{self.cube.cube_id}/?inline_no=15"
        )
        self.assertEqual(result.shape, (ninlines, nz))
        
    @patch('requests.get')
    def test_get_xline(self, mock_get):
        """Test get_xline method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Create sample data for trimmed=True
        nz, nxlines = 10, 5
        start_i = 3
        data = np.random.random((nxlines, nz)).astype(np.float32)
        header = struct.pack('<iii', start_i, nz, nxlines)
        mock_response.content = header + data.tobytes()
        
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Execute for trimmed=True
        result = self.cube.get_xline(15, trimmed=True, top_ind=2, bottom_ind=8)
        
        # Verify
        mock_get.assert_called_once_with(
            f"{self.cube.server_url}/seismic_3d/data/xline_trim/{self.cube.cube_id}/"
            f"?xline_no=15&top_no=2&bottom_no=8"
        )
        self.assertEqual(result.shape, (start_i + nxlines, nz))

    def test_generate_fragments_grid(self):
        """Test generate_fragments_grid method"""
        # Setup
        self.cube.min_i = 1
        self.cube.min_x = 1
        self.cube.n_i = 11
        self.cube.n_x = 6
        
        # Execute
        result = self.cube.generate_fragments_grid(2, 2)
        
        # Verify
        expected = [(1, 5, 1, 3), (1, 5, 4, 2), (6, 5, 1, 3), (6, 5, 4, 2)]
        self.assertEqual(result, expected)
        
    def test_generate_fragments_grid_incr(self):
        """Test generate_fragments_grid_incr method"""
        # Setup
        self.cube.min_i = 1
        self.cube.min_x = 1
        self.cube.n_i = 11
        self.cube.n_x = 6
        
        # Execute
        result = self.cube.generate_fragments_grid_incr(5, 2)
        
        # Verify
        expected = [(1, 5, 1, 2), (1, 5, 3, 2), (1, 5, 5, 1), (6, 5, 1, 2), (6, 5, 3, 2), (6, 5, 5, 1)]
        self.assertEqual(result, expected)
        
    @patch('requests.get')
    def test_get_statistics_for_horizons(self, mock_get):
        """Test get_statistics_for_horizons method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        stats_data = {"min": 10.5, "max": 100.2, "mean": 55.3, "std": 15.7}
        mock_response.content = json.dumps(stats_data).encode()
        
        mock_get.return_value.__enter__.return_value = mock_response
        
        # Execute
        result = self.cube.get_statistics_for_horizons("top_horizon", "bottom_horizon")
        
        # Verify
        mock_get.assert_called_once()
        self.assertEqual(result, stats_data)
        
    @patch('requests.post')
    def test_save_statistics_for_horizons(self, mock_post):
        """Test save_statistics_for_horizons method"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"status": "ok"}).encode()
        
        mock_post.return_value.__enter__.return_value = mock_response
        
        # Execute
        stats_data = {"min": 10.5, "max": 100.2, "mean": 55.3, "std": 15.7}
        self.cube.save_statistics_for_horizons("top_horizon", "bottom_horizon", stats_data)
        
        # Verify
        mock_post.assert_called_once()
        # Check that the data was properly prepared
        expected_data = {
            "min": 10.5, 
            "max": 100.2, 
            "mean": 55.3, 
            "std": 15.7,
            "hor_top": "top_horizon",
            "hor_bottom": "bottom_horizon"
        }
        called_args = mock_post.call_args[1]
        self.assertEqual(json.loads(called_args["data"].decode()), expected_data)


class TestDISeismicCubeWriter(unittest.TestCase):
    """Tests for the DISeismicCubeWriter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cube_writer = DISeismicCubeWriter(100, "test_name", "test_name2")
        self.cube_writer.server_url = "http://test.server"
        self.cube_writer.token = "test_token"
        
        # Sample cube info for testing
        self.cube_info = {
            "id": 123,
            "geometry_id": 456,
            "geometry_name": "test_geometry",
            "name": "test_name",
            "name2": "test_name2",
            "origin": [100, 200, 300],
            "d_inline": [1, 0, 0],
            "d_xline": [0, 1, 0],
            "max_inline": 100,
            "max_xline": 200,
            "nz": 300,
            "z_step": 4,
            "domain": "time",
            "z_start": 0,
            "min_inline": 1,
            "min_xline": 1
        }

    def test_init_from_info(self):
        """Test _init_from_info method"""
        # Execute
        self.cube_writer._init_from_info(self.cube_info)
        
        # Verify
        self.assertEqual(self.cube_writer.cube_id, 123)
        self.assertEqual(self.cube_writer.geometry_id, 456)
        self.assertEqual(self.cube_writer.geometry_name, "test_geometry")
        self.assertEqual(self.cube_writer.origin, [100, 200, 300])
        self.assertEqual(self.cube_writer.v_i, [1, 0, 0])
        self.assertEqual(self.cube_writer.v_x, [0, 1, 0])
        self.assertEqual(self.cube_writer.n_i, 99)  # Notice this is max_inline-1
        self.assertEqual(self.cube_writer.n_x, 199)  # Notice this is max_xline-1
        self.assertEqual(self.cube_writer.n_samples, 300)
        self.assertEqual(self.cube_writer.time_step, 4)
        self.assertEqual(self.cube_writer.domain, "time")
        self.assertEqual(self.cube_writer.data_start, 0)
        self.assertEqual(self.cube_writer.min_i, 1)
        self.assertEqual(self.cube_writer.min_x, 1)
        
    @patch('requests.post')
    def test_create(self, mock_post):
        """Test _create method"""
        # Setup
        self.cube_writer.geometry_id = 456
        self.cube_writer.n_i = 99
        self.cube_writer.n_x = 199
        self.cube_writer.n_samples = 300
        self.cube_writer.origin = [100, 200, 300]
        self.cube_writer.v_i = [1, 0, 0]
        self.cube_writer.v_x = [0, 1, 0]
        self.cube_writer.time_step = 4
        self.cube_writer.domain = "time"
        self.cube_writer.data_start = 0
        self.cube_writer.min_i = 1
        self.cube_writer.min_x = 1
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"id": 123}).encode()
        mock_post.return_value.__enter__.return_value = mock_response
        
        # Execute
        self.cube_writer._create()
        
        # Verify
        mock_post.assert_called_once()
        # Check the URL
        self.assertEqual(
            mock_post.call_args[0][0],
            f"{self.cube_writer.server_url}/seismic_3d/geometry/new_cube/{self.cube_writer.geometry_id}/"
        )
        # Check the cube ID was set
        self.assertEqual(self.cube_writer.cube_id, 123)
        
    @patch('requests.post')
    def test_write_fragment(self, mock_post):
        """Test write_fragment method"""
        # Setup
        self.cube_writer.cube_id = 123
        
        # Create sample data
        nz, ncdps, ninl = 10, 5, 3
        data = np.random.random((ninl, ncdps, nz)).astype(np.float32)
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value.__enter__.return_value = mock_response
        
        # Execute
        result = self.cube_writer.write_fragment(1, 5, data)
        
        # Verify
        mock_post.assert_called_once()
        # Check the URL
        self.assertEqual(
            mock_post.call_args[0][0],
            f"{self.cube_writer.server_url}/seismic_3d/data/rect_fragment/{self.cube_writer.cube_id}/?inline_no=1&xline_no=5"
        )
        # Check the headers
        self.assertEqual(
            mock_post.call_args[1]["headers"],
            {"Content-Type": "application/octet-stream"}
        )
        # Check that data was properly formatted
        called_data = mock_post.call_args[1]["data"]
        header = called_data[:12]
        unpacked_nz, unpacked_ncdps, unpacked_ninl = struct.unpack('<iii', header)
        self.assertEqual(unpacked_nz, nz)
        self.assertEqual(unpacked_ncdps, ncdps)
        self.assertEqual(unpacked_ninl, ninl)
        # The status code should be 200
        self.assertEqual(result, 200)


if __name__ == '__main__':
    unittest.main()
