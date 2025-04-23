import unittest
from unittest.mock import patch, MagicMock
import json
import requests
from di_lib.area import DIArea, AreaInfo, new_area

class TestAreaInfo(unittest.TestCase):
    def test_area_info_init(self):
        """Test basic AreaInfo initialization"""
        area_info = AreaInfo(name="test_area", area=[(1.0, 2.0), (3.0, 4.0)], ts="2025-03-05")
        self.assertEqual(area_info.name, "test_area")
        self.assertIsNone(area_info.id)
        self.assertEqual(area_info.area, [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(area_info.ts, "2025-03-05")
        self.assertIsNone(area_info.user_name)
    
    def test_ensure_list_validator(self):
        """Test that the ensure_list validator correctly handles None values"""
        area_info = AreaInfo(name="test_area", area=None, ts="2025-03-05")
        self.assertEqual(area_info.area, [])

class TestDIArea(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.project_id = 123
        self.area_name = "test_area"
        self.area = DIArea(self.project_id, self.area_name)
        self.area.server_url = "http://test-server.com"
        self.area.token = "test-token"
        
        # Sample polygon data
        self.polygon_data = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]
        
    @patch('requests.get')
    def test_read_info(self, mock_get):
        """Test the _read_info method with mocked API responses"""
        # Mock the first request to get area list
        mock_areas_response = MagicMock()
        mock_areas_response.status_code = 200
        mock_areas_response.json.return_value = [
            {"id": 456, "name": "test_area"},
            {"id": 789, "name": "other_area"}
        ]
        
        # Mock the second request to get area properties
        mock_props_response = MagicMock()
        mock_props_response.status_code = 200
        mock_props_response.json.return_value = {
            "id": 456,
            "name": "test_area",
            "area": self.polygon_data,
            "ts": "2025-03-05T12:00:00Z",
            "owner": "test_user"
        }
        
        # Configure the mock to return different responses for different URLs
        mock_get.side_effect = [mock_areas_response, mock_props_response]
        
        # Call the method
        area_info = self.area._read_info()
        
        # Check that requests were made with correct URLs
        expected_urls = [
            f"{self.area.server_url}/grids/areas/list/{self.project_id}/",
            f"{self.area.server_url}/grids/areas/properties/{self.project_id}/456/"
        ]
        actual_urls = [call.args[0] for call in mock_get.call_args_list]
        self.assertEqual(actual_urls, expected_urls)
        
        # Check the returned AreaInfo object
        self.assertEqual(area_info.id, 456)
        self.assertEqual(area_info.name, "test_area")
        self.assertEqual(area_info.area, self.polygon_data)
        self.assertEqual(area_info.ts, "2025-03-05T12:00:00Z")
        self.assertEqual(area_info.user_name, "test_user")
    
    @patch('requests.get')
    def test_read_info_area_not_found(self, mock_get):
        """Test _read_info when the area is not found"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 789, "name": "other_area"}
        ]
        
        mock_get.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            self.area._read_info()
        
        self.assertIn(f"Area {self.area_name} not found in project {self.project_id}", str(context.exception))
    
    @patch('requests.get')
    def test_read_info_request_failure(self, mock_get):
        """Test _read_info when the API request fails"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b"Server error"
        
        mock_get.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            self.area._read_info()
        
        self.assertIn("Can't get list of areas", str(context.exception))
    
    @patch('di_lib.area.DIArea._read_info')
    def test_polygon_getter(self, mock_read_info):
        """Test the polygon property getter"""
        mock_area_info = AreaInfo(
            id=456,
            name="test_area",
            area=self.polygon_data,
            ts="2025-03-05T12:00:00Z",
            user_name="test_user"
        )
        mock_read_info.return_value = mock_area_info
        
        # Set id to None to trigger _read_info call
        self.area._area_info.id = None
        
        # Call polygon getter
        polygon = self.area.polygon
        
        # Check that _read_info was called
        mock_read_info.assert_called_once()
        
        # Check that the correct polygon was returned
        self.assertEqual(polygon, self.polygon_data)
    
    @patch('di_lib.area.DIArea._update')
    def test_polygon_setter(self, mock_update):
        """Test the polygon property setter"""
        new_polygon = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (10.0, 20.0)]
        
        # Call polygon setter
        self.area.polygon = new_polygon
        
        # Check that the polygon was updated
        self.assertEqual(self.area._area_info.area, new_polygon)
        
        # Check that _update was called
        mock_update.assert_called_once()
    
    @patch('requests.post')
    def test_update(self, mock_post):
        """Test the _update method"""
        # Create a mock for the context manager
        mock_context = MagicMock()
        mock_context.status_code = 200
        mock_context.content = json.dumps({
            "id": 456,
            "ts": "2025-03-05T12:00:00Z"
        }).encode("utf8")
        
        # Configure mock_post to return the context manager mock
        mock_post.return_value.__enter__.return_value = mock_context
        
        # Set up area info
        self.area._area_info = AreaInfo(
            name="test_area",
            area=self.polygon_data,
            ts="old_timestamp",
            user_name="test_user"
        )
        
        # Call _update
        self.area._update()
        
        # Check that the request was made with correct parameters
        mock_post.assert_called_once()
        url = f"{self.area.server_url}/grids/areas/update/{self.project_id}/"
        self.assertEqual(mock_post.call_args[0][0], url)
        
        # Check headers
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["x-di-authorization"], "test-token")
        
        # Check request body - taking into account JSON serialization of tuples as lists
        expected_body_dict = {
            "name": "test_area",
            "area": [[point[0], point[1]] for point in self.polygon_data]  # Convert tuples to lists
        }
        actual_body = json.loads(mock_post.call_args[1]["data"].decode("utf8"))
        self.assertEqual(actual_body, expected_body_dict)
        
        # Check that area_info was updated
        self.assertEqual(self.area._area_info.id, 456)
        self.assertEqual(self.area._area_info.ts, "2025-03-05T12:00:00Z")
    
    @patch('requests.post')
    def test_update_failure(self, mock_post):
        """Test _update when the API request fails"""
        # Create a mock for the context manager
        mock_context = MagicMock()
        mock_context.status_code = 500
        mock_context.content = b"Server error"
        
        # Configure mock_post to return the context manager mock
        mock_post.return_value.__enter__.return_value = mock_context
        
        self.area._area_info = AreaInfo(
            name="test_area",
            area=self.polygon_data,
            ts="",
            user_name=None
        )
        
        with self.assertRaises(RuntimeError) as context:
            self.area._update()
        
        self.assertIn("Failed to update area", str(context.exception))
    
    @patch('requests.post')
    def test_create(self, mock_post):
        """Test the _create method"""
        # Create a mock for the context manager
        mock_context = MagicMock()
        mock_context.status_code = 200
        mock_context.content = json.dumps({
            "id": 456,
            "ts": "2025-03-05T12:00:00Z"
        }).encode("utf8")
        
        # Configure mock_post to return the context manager mock
        mock_post.return_value.__enter__.return_value = mock_context
        
        # Set up area info
        self.area._area_info = AreaInfo(
            name="test_area",
            area=self.polygon_data,
            ts="",
            user_name=None
        )
        
        # Call _create
        self.area._create()
        
        # Check that the request was made with correct parameters
        mock_post.assert_called_once()
        url = f"{self.area.server_url}/grids/areas/update/{self.project_id}/"
        self.assertEqual(mock_post.call_args[0][0], url)
        
        # Check headers
        headers = mock_post.call_args[1]["headers"]
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["x-di-authorization"], "test-token")
        
        # Check request body - taking into account JSON serialization of tuples as lists
        expected_body_dict = {
            "name": "test_area",
            "area": [[point[0], point[1]] for point in self.polygon_data]  # Convert tuples to lists
        }
        actual_body = json.loads(mock_post.call_args[1]["data"].decode("utf8"))
        self.assertEqual(actual_body, expected_body_dict)
        
        # Check that area_info was updated
        self.assertEqual(self.area._area_info.id, 456)
        self.assertEqual(self.area._area_info.ts, "2025-03-05T12:00:00Z")
    
    @patch('requests.post')
    def test_create_failure(self, mock_post):
        """Test _create when the API request fails"""
        # Create a mock for the context manager
        mock_context = MagicMock()
        mock_context.status_code = 500
        mock_context.content = b"Server error"
        
        # Configure mock_post to return the context manager mock
        mock_post.return_value.__enter__.return_value = mock_context
        
        self.area._area_info = AreaInfo(
            name="test_area",
            area=self.polygon_data,
            ts="",
            user_name=None
        )
        
        with self.assertRaises(RuntimeError) as context:
            self.area._create()
        
        self.assertIn("Failed to update area", str(context.exception))

class TestNewArea(unittest.TestCase):
    @patch('di_lib.area.DIArea._create')
    def test_new_area(self, mock_create):
        """Test the new_area function"""
        server_url = "http://test-server.com"
        token = "test-token"
        project_id = 123
        name = "new_test_area"
        path = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (1.0, 2.0)]
        
        # Call new_area
        area = new_area(server_url, token, project_id, name, path)
        
        # Check that DIArea was initialized correctly
        self.assertEqual(area.server_url, server_url)
        self.assertEqual(area.token, token)
        self.assertEqual(area.project_id, project_id)
        self.assertEqual(area.name, name)
        
        # Check that AreaInfo was initialized correctly
        self.assertEqual(area._area_info.name, name)
        self.assertIsNone(area._area_info.id)
        self.assertEqual(area._area_info.area, path)
        self.assertEqual(area._area_info.ts, "")
        self.assertIsNone(area._area_info.user_name)
        
        # Check that _create was called
        mock_create.assert_called_once()

if __name__ == '__main__':
    unittest.main()
