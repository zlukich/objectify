from flask import Flask, request, jsonify

import sys
import os
# Add the path to 'lib' directory
sys.path.append(os.path.abspath(os.path.join('lib')))

from config.ConfigManager import ConfigManager  # Assuming the ConfigManager class is in the same directory

app = Flask(__name__)

# Path to your configuration file
config_file_path = 'node-red/config.json'
config_manager = ConfigManager(config_file_path)

@app.route('/config/<project_name>', methods=['GET', 'POST', 'DELETE'])
def handle_project(project_name):
    print("test")
    if request.method == 'GET':
        project_data = config_manager.get_project(project_name)
        print("like project", project_data)
        return jsonify(project_data)
    elif request.method == 'POST':
        data = request.json
        config_manager.update_project(project_name, data)
        return jsonify({"status": "success"})
    elif request.method == 'DELETE':
        config_manager.delete_project(project_name)
        return jsonify({"status": "deleted"})

@app.route('/config/current_work', methods=['GET', 'POST', 'DELETE'])
def handle_current_work():
    print("test")
    if request.method == 'GET':
        current_work_data = config_manager.get_current_work()
        print("like current_work", current_work_data)

        return jsonify(current_work_data)
    elif request.method == 'POST':
        data = request.json
        config_manager.update_current_work(data)
        return jsonify({"status": "success"})
    elif request.method == 'DELETE':
        config_manager.clear_current_work()
        return jsonify({"status": "current_work cleared"})

if __name__ == '__main__':
    app.run(port=5001, host='127.0.0.1')
