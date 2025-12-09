import os
import time
import json
import logging
import board
import adafruit_dht
import paho.mqtt.client as mqtt
from collections import deque
from datetime import datetime

class PlantSensorMonitor:
    """Reads plant sensors and publishes to MQTT"""
    
    def __init__(self, mqtt_broker, mqtt_port=1883, reading_interval=30):
        """
        Initialize sensor monitor
        
        Args:
            mqtt_broker: IP or hostname of MQTT broker (laptop)
            mqtt_port: MQTT broker port
            reading_interval: Seconds between readings
        """
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.reading_interval = reading_interval
        self.mqtt_client = None
        self.is_connected = False
        
        self.dht_device = None
        self.soil_sensor_pin = None
        self.ldr_pin = None
        
        self.last_readings = deque(maxlen=100)
        self.reading_count = 0
        
    
    def setup_sensors(self):
        """Initialize sensor connections"""
        try:
            self.dht_device = adafruit_dht.DHT22(board.D17)
            
            return True
            
        except Exception as e:
            return False
    
    def setup_mqtt(self):
        """Setup MQTT connection"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
            self.mqtt_client.on_publish = self.on_mqtt_publish
            
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
            
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.is_connected = True
        else:
            self.is_connected = False
    
    def on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.is_connected = False
    
    def on_mqtt_publish(self, client, userdata, mid):
        """MQTT publish callback"""
    
    def read_dht22(self):
        """Read temperature and humidity from DHT22"""
        try:
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            
            if temperature is not None and humidity is not None:
                return {
                    'temperature': round(temperature, 2),
                    'humidity': round(humidity, 2)
                }
            else:
                return {'temperature': None, 'humidity': None}
                
        except RuntimeError as e:
            return {'temperature': None, 'humidity': None}
        except Exception as e:
            return {'temperature': None, 'humidity': None}
    
    def read_soil_moisture(self):
        """
        Read capacitive soil moisture sensor
        
        Note: This requires an ADC (Analog-to-Digital Converter) like MCP3008
        connected via SPI. Adjust pin numbers based on your setup.
        
        For running without ADC, this returns simulated values.
        """
        try:
            try:
                import busio
                import digitalio
                from adafruit_mcp230xx.digital_inout import DigitalInOut
                from adafruit_mcp3008 import MCP3008
                
                spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
                cs = digitalio.DigitalInOut(board.D8)
                mcp = MCP3008(spi, cs)
                
                raw_value = mcp.channel[0].value
                
                soil_moisture = (raw_value / 1023.0) * 100
                
                return {
                    'soil_moisture': round(soil_moisture, 2),
                    'soil_raw': raw_value
                }
            except (ImportError, Exception) as e:
                
                import random
                simulated_moisture = round(random.uniform(20, 80), 2)
                return {
                    'soil_moisture': simulated_moisture,
                    'soil_raw': int(simulated_moisture * 10.23)
                }
                
        except Exception as e:
            return {'soil_moisture': None, 'soil_raw': None}
    
    def read_light_level(self):
        """
        Read light level from LDR
        
        Note: This requires an ADC like MCP3008.
        For running without ADC, returns simulated values.
        """
        try:
            try:
                import busio
                import digitalio
                from adafruit_mcp3008 import MCP3008
                
                spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
                cs = digitalio.DigitalInOut(board.D8)
                mcp = MCP3008(spi, cs)
                
                raw_value = mcp.channel[1].value
                
                light_level = (raw_value / 1023.0) * 100
                
                return {
                    'light_level': round(light_level, 2),
                    'light_raw': raw_value
                }
            except (ImportError, Exception) as e:
                
                import random
                simulated_light = round(random.uniform(10, 100), 2)
                return {
                    'light_level': simulated_light,
                    'light_raw': int(simulated_light * 10.23)
                }
                
        except Exception as e:
            return {'light_level': None, 'light_raw': None}
    
    def read_all_sensors(self):
        """Read all sensors and return data"""
        reading = {
            'timestamp': datetime.now().isoformat(),
            'reading_id': self.reading_count
        }
        
        dht_data = self.read_dht22()
        reading.update(dht_data)
        
        soil_data = self.read_soil_moisture()
        reading.update(soil_data)
        
        light_data = self.read_light_level()
        reading.update(light_data)
        
        self.reading_count += 1
        self.last_readings.append(reading)
        
        return reading
    
    def publish_reading(self, reading):
        """Publish sensor reading to MQTT"""
        if not self.is_connected:
            return False
        
        try:
            payload = json.dumps(reading)
            
            result = self.mqtt_client.publish(
                topic="plant/sensors/raw",
                payload=payload,
                qos=1
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def start_monitoring(self):
        """Start continuous sensor monitoring"""
        try:
            if not self.setup_sensors():
                return
            
            if not self.setup_mqtt():
                return
            
            while True:
                try:
                    reading = self.read_all_sensors()

                    self._print_reading(reading)

                    self.publish_reading(reading)

                    time.sleep(self.reading_interval)
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    time.sleep(5) 

        finally:
            self.cleanup()
    
    def _print_reading(self, reading):
        """Pretty-print sensor reading"""
        print(f"\n[{reading['timestamp']}] Reading #{reading['reading_id']}")
        print("─" * 50)
        
        if reading.get('temperature') is not None:
            print(f"Temperature: {reading['temperature']}°C")
        if reading.get('humidity') is not None:
            print(f"Humidity: {reading['humidity']}%")
        if reading.get('soil_moisture') is not None:
            print(f"Soil Moisture: {reading['soil_moisture']}%")
        if reading.get('light_level') is not None:
            print(f"Light Level: {reading['light_level']}%")
    
    def cleanup(self):
        """Cleanup resources"""
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if self.dht_device:
            self.dht_device.deinit()
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plant Monitor - Raspberry Pi Sensor Reader')
    parser.add_argument(
        '--broker',
        required=True,
        help='MQTT broker IP/hostname (e.g., 192.168.1.100 or laptop.local)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=1883,
        help='MQTT broker port (default: 1883)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Reading interval in seconds (default: 30)'
    )
    
    args = parser.parse_args()

    print(f"\nStarting in 5 seconds...\n")
    
    time.sleep(5)
    
    monitor = PlantSensorMonitor(args.broker, args.port, args.interval)
    monitor.start_monitoring()
