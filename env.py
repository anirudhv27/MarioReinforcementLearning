import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Set up the Super Mario Bros gym environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env.reset()

# Server for MJPEG streaming
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()

        done = True
        for step in range(5000):
            if done:
                state = env.reset()
            state, reward, done, info = env.step(env.action_space.sample())
            img = env.render(mode="rgb_array")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            _, jpeg = cv2.imencode('.jpg', img)
            self.wfile.write("--jpgboundary".encode())
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-length', str(jpeg.size))
            self.end_headers()
            self.wfile.write(jpeg.tobytes())

server = HTTPServer(('0.0.0.0', 8080), MJPEGHandler)

def server_thread():
    server.serve_forever()

t = threading.Thread(target=server_thread)
t.start()

print("Server started at http://localhost:8080")

try:
    while True:
        pass
except KeyboardInterrupt:
    server.shutdown()
    env.close()
