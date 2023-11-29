import unittest
import yaml
from unittest.mock import patch, mock_open
from src.utils.read_config import read_config

class TestReadConfig(unittest.TestCase):
    def setUp(self):
        self.mock_config_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
        }
        self.config_path = "config.yaml"

    def test_read_config(self):
        mock_config_yaml = yaml.dump(self.mock_config_data)
        with patch("builtins.open", new_callable=mock_open, read_data=mock_config_yaml) as mock_file:
            config = read_config(self.config_path)
            self.assertEqual(config, self.mock_config_data)
            mock_file.assert_called_once_with(self.config_path, "r")

    def test_read_config_file_not_found(self):
        with patch("builtins.open", side_effect=FileNotFoundError()) as mock_file:
            with self.assertRaises(FileNotFoundError) as context:
                read_config(self.config_path)
            self.assertEqual(str(context.exception), f"Error: Config file not found at '{self.config_path}'")
            mock_file.assert_called_once_with(self.config_path, "r")

if __name__ == "__main__":
    unittest.main()