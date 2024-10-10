import requests

class ConfigManagerAPI:
    def __init__(self, base_url):
        """
        Initialize the ConfigManagerAPI class.

        :param base_url: Base URL of the ConfigManager Flask server (e.g., "http://localhost:5001").
        """
        self.base_url = base_url

    def get_project(self, project_name):
        """
        Get the configuration for a specific project.

        :param project_name: Name of the project.
        :return: JSON response containing the project data.
        """
        url = f"{self.base_url}/config/{project_name}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    def get_config(self):
        """
        Get the configuration for a specific project.

        :param project_name: Name of the project.
        :return: JSON response containing the project data.
        """
        url = f"{self.base_url}/config/"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    def update_project(self, project_name, data):
        """
        Update the configuration for a specific project.

        :param project_name: Name of the project.
        :param data: Dictionary containing the data to update.
        :return: JSON response indicating success or failure.
        """
        url = f"{self.base_url}/config/{project_name}"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def delete_project(self, project_name):
        """
        Delete a specific project from the configuration.

        :param project_name: Name of the project.
        :return: JSON response indicating success or failure.
        """
        url = f"{self.base_url}/config/{project_name}"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()

    def get_current_work(self):
        """
        Get the current work parameters from the configuration.

        :return: JSON response containing the current work parameters.
        """
        url = f"{self.base_url}/config/current_work"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def update_current_work(self, data):
        """
        Update the current work parameters in the configuration.

        :param data: Dictionary containing the current work parameters to update.
        :return: JSON response indicating success or failure.
        """
        url = f"{self.base_url}/config/current_work"
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def clear_current_work(self):
        """
        Clear the current work parameters from the configuration.

        :return: JSON response indicating success or failure.
        """
        url = f"{self.base_url}/config/current_work"
        response = requests.delete(url)
        response.raise_for_status()
        return response.json()