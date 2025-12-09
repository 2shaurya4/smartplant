"""
Plant Monitor - Laptop MQTT Subscriber & ML Server
Receives sensor data from RPi, runs ML inference, stores results
Provides REST API for Freeboard dashboard
"""

import paho.mqtt.client as mqtt
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

from plant_ml_classifier import PlantWateringClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

plant_monitor = {
    'mqtt_client': None,
    'mqtt_connected': False,
    'classifier': None,
    'classifier_ready': False,
    

    'raw_readings': deque(maxlen=200),  
    'ml_predictions': deque(maxlen=200),  
    
    'total_readings': 0,
    'watering_recommendations_count': {
        0: 0,  
        1: 0,  
        2: 0   
    },
    
    'last_raw_reading': None,
    'last_prediction': None,
    'current_status': 'initializing'
}


def setup_classifier():
    """Initialize ML classifier"""
    try:
        classifier = PlantWateringClassifier(model_type='decision_tree')
        
        if os.path.exists('plant_watering_model.pkl'):
            classifier.load_model('plant_watering_model.pkl')
        else:

            classifier.train()
            classifier.save_model('plant_watering_model.pkl')
        
        plant_monitor['classifier'] = classifier
        plant_monitor['classifier_ready'] = True
        return True
        
    except Exception as e:
        plant_monitor['classifier_ready'] = False
        return False


def setup_mqtt(broker='localhost', port=1883):
    """Setup MQTT connection"""
    try:
        client = mqtt.Client()
        client.on_connect = on_mqtt_connect
        client.on_disconnect = on_mqtt_disconnect
        client.on_message = on_mqtt_message
        
        client.connect(broker, port, keepalive=60)
        client.subscribe('plant/sensors/raw')
        
        client.loop_start()
        
        plant_monitor['mqtt_client'] = client

        timeout = 10
        start_time = time.time()
        while not plant_monitor['mqtt_connected'] and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if plant_monitor['mqtt_connected']:
            return True
        else:
            return False
            
    except Exception as e:
        return False


def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        plant_monitor['mqtt_connected'] = True
        plant_monitor['current_status'] = 'connected'
        client.subscribe('plant/sensors/raw')
    else:
        plant_monitor['mqtt_connected'] = False


def on_mqtt_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""

    plant_monitor['mqtt_connected'] = False
    plant_monitor['current_status'] = 'disconnected'


def on_mqtt_message(client, userdata, msg):
    """MQTT message received callback"""
    payload = json.loads(msg.payload.decode())

    plant_monitor['raw_readings'].append(payload)
    plant_monitor['last_raw_reading'] = payload
    plant_monitor['total_readings'] += 1

    if plant_monitor['classifier_ready']:
        run_inference(payload)
        


def run_inference(sensor_reading):
    """Run ML inference on sensor reading"""
    classifier = plant_monitor['classifier']
        
    prediction = classifier.predict(sensor_reading)

    prediction['reading_id'] = sensor_reading.get('reading_id')

    plant_monitor['ml_predictions'].append(prediction)
    plant_monitor['last_prediction'] = prediction

    pred_class = prediction['prediction']
    plant_monitor['watering_recommendations_count'][pred_class] += 1


# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'mqtt_connected': plant_monitor['mqtt_connected'],
        'classifier_ready': plant_monitor['classifier_ready']
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'system_status': plant_monitor['current_status'],
        'mqtt_connected': plant_monitor['mqtt_connected'],
        'classifier_ready': plant_monitor['classifier_ready'],
        'total_readings': plant_monitor['total_readings'],
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/latest', methods=['GET'])
def get_latest():
    """Get latest raw and predicted data"""
    result = {
        'raw_reading': plant_monitor['last_raw_reading'],
        'prediction': plant_monitor['last_prediction'],
        'timestamp': datetime.now().isoformat()
    }
    
    if plant_monitor['last_prediction']:
        pred = plant_monitor['last_prediction']
        result['recommendation_text'] = pred['recommendation']
        result['confidence_percent'] = round(pred['confidence'] * 100, 1)
    
    return jsonify(result), 200


@app.route('/api/raw-readings', methods=['GET'])
def get_raw_readings():
    """Get raw sensor readings history"""
    limit = request.args.get('limit', 50, type=int)
    readings = list(plant_monitor['raw_readings'])[-limit:]
    
    return jsonify({
        'count': len(readings),
        'readings': readings
    }), 200


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get ML predictions history"""
    limit = request.args.get('limit', 50, type=int)
    predictions = list(plant_monitor['ml_predictions'])[-limit:]
    
    return jsonify({
        'count': len(predictions),
        'predictions': predictions
    }), 200


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    total = sum(plant_monitor['watering_recommendations_count'].values())
    
    stats = {
        'total_readings': plant_monitor['total_readings'],
        'total_predictions': total,
        'recommendations': {
            "don't_water": plant_monitor['watering_recommendations_count'][0],
            'water_soon': plant_monitor['watering_recommendations_count'][1],
            'water_now': plant_monitor['watering_recommendations_count'][2]
        }
    }
    
    if total > 0:
        stats['recommendation_percentages'] = {
            "don't_water": round(
                (plant_monitor['watering_recommendations_count'][0] / total) * 100, 1
            ),
            'water_soon': round(
                (plant_monitor['watering_recommendations_count'][1] / total) * 100, 1
            ),
            'water_now': round(
                (plant_monitor['watering_recommendations_count'][2] / total) * 100, 1
            )
        }
    
    return jsonify(stats), 200


@app.route('/api/sensor-metrics', methods=['GET'])
def get_sensor_metrics():
    """Get current sensor metrics"""
    if not plant_monitor['last_raw_reading']:
        return jsonify({'error': 'No data yet'}), 202
    
    reading = plant_monitor['last_raw_reading']
    
    return jsonify({
        'timestamp': reading.get('timestamp'),
        'soil_moisture': reading.get('soil_moisture'),
        'temperature': reading.get('temperature'),
        'humidity': reading.get('humidity'),
        'light_level': reading.get('light_level')
    }), 200


@app.route('/api/recommendation', methods=['GET'])
def get_recommendation():
    """Get current watering recommendation"""
    if not plant_monitor['last_prediction']:
        return jsonify({'error': 'No prediction yet'}), 202
    
    pred = plant_monitor['last_prediction']
    
    return jsonify({
        'timestamp': pred.get('timestamp'),
        'recommendation': pred.get('recommendation'),
        'confidence': round(pred.get('confidence', 0) * 100, 1),
        'prediction_id': pred.get('prediction'),
        'probabilities': pred.get('probabilities')
    }), 200


@app.route('/api/detailed', methods=['GET'])
def get_detailed():
    """Get combined detailed view for dashboard"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'mqtt_status': 'connected' if plant_monitor['mqtt_connected'] else 'disconnected',
        'classifier_status': 'ready' if plant_monitor['classifier_ready'] else 'not_ready'
    }
    
    if plant_monitor['last_raw_reading']:
        reading = plant_monitor['last_raw_reading']
        result['sensors'] = {
            'soil_moisture': reading.get('soil_moisture'),
            'temperature': reading.get('temperature'),
            'humidity': reading.get('humidity'),
            'light_level': reading.get('light_level'),
            'timestamp': reading.get('timestamp')
        }
    
    if plant_monitor['last_prediction']:
        pred = plant_monitor['last_prediction']
        result['recommendation'] = {
            'text': pred.get('recommendation'),
            'confidence': round(pred.get('confidence', 0) * 100, 1),
            'all_probabilities': {
                k: round(v * 100, 1) 
                for k, v in pred.get('probabilities', {}).items()
            }
        }
    
    return jsonify(result), 200


@app.route('/api/history-combined', methods=['GET'])
def get_history_combined():
    """Get combined history of readings and predictions"""
    limit = request.args.get('limit', 100, type=int)
    
    readings = list(plant_monitor['raw_readings'])[-limit:]
    predictions = list(plant_monitor['ml_predictions'])[-limit:]
    
    combined = []
    for i, reading in enumerate(readings):
        item = {
            'reading_id': reading.get('reading_id'),
            'timestamp': reading.get('timestamp'),
            'sensors': {
                'soil_moisture': reading.get('soil_moisture'),
                'temperature': reading.get('temperature'),
                'humidity': reading.get('humidity'),
                'light_level': reading.get('light_level')
            }
        }
        
        if i < len(predictions):
            pred = predictions[i]
            item['recommendation'] = {
                'text': pred.get('recommendation'),
                'confidence': round(pred.get('confidence', 0) * 100, 1)
            }
        
        combined.append(item)
    
    return jsonify({
        'count': len(combined),
        'history': combined
    }), 200


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Monitor - Laptop Server')
    parser.add_argument(
        '--broker',
        default='localhost',
        help='MQTT broker IP/hostname (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Flask server port (default: 5000)'
    )
    parser.add_argument(
        '--mqtt-port',
        type=int,
        default=1883,
        help='MQTT broker port (default: 1883)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Plant Monitor - Laptop Server (MQTT Subscriber + ML Inference)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  MQTT Broker: {args.broker}:{args.mqtt_port}")
    print(f"  Flask API: http://0.0.0.0:{args.port}")
    print("\nStarting components...\n")
    
    print("1. Loading ML Classifier...")
    if not setup_classifier():
        return
    
    print("\n2. Connecting to MQTT Broker...")
    if not setup_mqtt(args.broker, args.mqtt_port):
        return
    
    print("\n3. Starting Flask API Server...")
    print(f"   API Documentation: http://localhost:{args.port}")
    print("\n" + "=" * 70)
    print("System ready! Waiting for sensor readings from Raspberry Pi...")
    print("=" * 70 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    finally:
        if plant_monitor['mqtt_client']:
            plant_monitor['mqtt_client'].loop_stop()
            plant_monitor['mqtt_client'].disconnect()


if __name__ == '__main__':
    main()
