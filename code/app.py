import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from inference import get_category

app = Flask(__name__)
CORS(app)

#Cargar configuraciones de config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
# Obtener valores de configuración
flask_port = config.get("flask_port_server")

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    if 'imagen_base64' in data:
        imagen_base64 = data['imagen_base64']

    imagen = base64.b64decode(imagen_base64)

    data = get_category(imagen, data['tipo_modelo'])

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=flask_port)