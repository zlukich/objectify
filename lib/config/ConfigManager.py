import json
import os
import tempfile

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        if not os.path.exists(self.config_file):
            return {}
        try:
            with open(self.config_file, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error loading config file: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading config file: {e}")
            return {}

    def _save_config(self, data):
        """Save configuration to file using a temporary file for atomic writes."""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8')
            with temp_file as file:
                json.dump(data, file, indent=4)
            # Rename temporary file to the actual config file
            os.replace(temp_file.name, self.config_file)
        except (IOError, OSError) as e:
            logging.error(f"Error saving config file: {e}")
        finally:
            # Clean up the temp file if it still exists
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def delete_project(self, project_name):
        """Delete a project from the configuration."""
        try:
            if project_name in self.config_data:
                del self.config_data[project_name]
                self._save_config(self.config_data)
                logging.info(f"Project '{project_name}' deleted successfully.")
            else:
                logging.warning(f"Project '{project_name}' not found.")
        except Exception as e:
            logging.error(f"Error deleting project '{project_name}': {e}")

    def update_project(self, project_name, data):
        """Update project configuration and save it."""
        try:
            if project_name not in self.config_data:
                self.config_data[project_name] = {}
            self.config_data[project_name].update(data)
            self._save_config(self.config_data)
        except Exception as e:
            logging.error(f"Error updating project '{project_name}': {e}")

    def get_project(self, project_name):
        return self.config_data.get(project_name, {})