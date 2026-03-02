class ESP32Receiver:
    def __init__(self, source="esp32", ip="192.168.1.6", port=81, serial_port="COM3", baud=115200):
        self._running = False
    def _serial_loop(self):
        pass
    def get_distance(self):
        return None
