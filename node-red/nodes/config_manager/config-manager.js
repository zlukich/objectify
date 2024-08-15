module.exports = function(RED) {
    function ConfigManagerNode(config) {
        RED.nodes.createNode(this, config);
        var node = this;
        const axios = require('axios');

        node.on('input', async function(msg) {
            const action = config.action || msg.payload.action;
            const project_name = config.project || msg.payload.project;
            const data = msg.payload.data || {};

            let url = `http://localhost:5001/config/${project_name}`;
            try {
                let response;
                if (action === 'get') {
                    response = await axios.get(url);
                } else if (action === 'update') {
                    response = await axios.post(url, data);
                } else if (action === 'delete') {
                    response = await axios.delete(url);
                }
                msg.payload = response.data;
                node.send(msg);
            } catch (error) {
                node.error(`Error: ${error.message}`);
            }
        });
    }
    RED.nodes.registerType('config-manager', ConfigManagerNode);
};

