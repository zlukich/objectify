# config_server.py
from flask import Flask, request, jsonify
from ConfigManager import ConfigManager  # Your ConfigManager class

app = Flask(__name__)
config_manager = ConfigManager('scripts/config.json')

@app.route('/config/<project_name>', methods=['GET', 'POST'])
def handle_config(project_name):
    if request.method == 'POST':
        data = request.json
        config_manager.update_project(project_name, data)
        return jsonify({"status": "success"})
    elif request.method == 'GET':
        project_data = config_manager.get_project(project_name)
        return jsonify(project_data)
    
@app.route('/config/current_work', methods=['GET', 'POST', 'DELETE'])
def handle_current_work():
    if request.method == 'POST':
        data = request.json
        config_manager.update_current_work(data)
        return jsonify({"status": "success"})
    elif request.method == 'GET':
        current_work_data = config_manager.get_current_work()
        return jsonify(current_work_data)
    elif request.method == 'DELETE':
        config_manager.clear_current_work()
        return jsonify({"status": "current_work cleared"})


if __name__ == '__main__':
    app.run(port=5001)
