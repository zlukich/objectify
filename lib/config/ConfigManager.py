import json
import os
import tempfile

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self._load_config()
        self._initialize_current_work()

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

    def _save_config(self):
        """Save configuration to file using a temporary file for atomic writes."""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='', encoding='utf-8')
            with temp_file as file:
                json.dump(self.config_data, file, indent=4)
            os.replace(temp_file.name, self.config_file)
        except (IOError, OSError) as e:
            logging.error(f"Error saving config file: {e}")
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    def _initialize_current_work(self):
        """Initialize or clear the 'current_work' section of the config."""
        self.config_data['current_work'] = {}
        self._save_config()

    def update_project(self, project_name, data):
        """Update project configuration and save it."""
        try:
            if project_name not in self.config_data:
                self.config_data[project_name] = {}
            self.config_data[project_name].update(data)
            self._save_config()
        except Exception as e:
            logging.error(f"Error updating project '{project_name}': {e}")

    def get_project(self, project_name):
        """Retrieve project configuration."""
        return self.config_data.get(project_name, {})

    def delete_project(self, project_name):
        """Delete a project from the configuration."""
        try:
            if project_name in self.config_data:
                del self.config_data[project_name]
                self._save_config()
                logging.info(f"Project '{project_name}' deleted successfully.")
            else:
                logging.warning(f"Project '{project_name}' not found.")
        except Exception as e:
            logging.error(f"Error deleting project '{project_name}': {e}")

    def update_current_work(self, data):
        """Update 'current_work' section with new data."""
        try:
            self.config_data['current_work'] = data
            self._save_config()
        except Exception as e:
            logging.error(f"Error updating 'current_work': {e}")

    def get_current_work(self):
        """Retrieve the current work configuration."""
        return self.config_data.get('current_work', {})

    def clear_current_work(self):
        """Clear the 'current_work' section."""
        self.update_current_work({})