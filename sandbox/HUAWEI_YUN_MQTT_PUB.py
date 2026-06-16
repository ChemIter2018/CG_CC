import paho.mqtt.client as mqtt
import time

def on_connect(mqttc, obj, flags, rc):
    print("Connected with result code " + str(rc))

def on_message(mqttc, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

mqttc = mqtt.Client()
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect("121.36.231.11", 1883, 60)
mqttc.loop_start()

sum = 0

while True:
    sum += 1
    print(sum)
    if sum >= 100:
        sum = 0

    mqttc.publish("message", payload=str(sum))
    time.sleep(1)