from pynput import keyboard
import time

class KeyListener:
    def __init__(self, key_states, callbackFunc=None):
        self.listener = keyboard.Listener(on_press=self.onPress, on_release=self.onRelease)
        self.key_states = key_states
        self.callbackFunc = callbackFunc

    def join(self):
        self.listener.join()

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def onPress(self, key):
        if key in self.key_states:
            self.key_states[key] = True

        if key == keyboard.Key.esc:
            return False

    def onRelease(self, key):
        if key in self.key_states:
            self.key_states[key] = False

    def callbackFunc(self):
        pass

def test_callback():
    print('test_callback')

if __name__ == '__main__':
    key_states = {
        keyboard.Key.up: False,
        keyboard.Key.down: False,
        keyboard.Key.left: False,
        keyboard.Key.right: False,
        keyboard.Key.alt_l: False,
        keyboard.Key.alt_r: False,
    }
    key_listener = KeyListener(key_states, callbackFunc=test_callback)
    key_listener.start()

    import threading,tkinter as tk
    def startGui():
        root = tk.Tk()
        root.title("WSL2 X11 Bridge")
        root.geometry("1x1")
        root.iconify()
        root.mainloop()

    def runGuiThread():
        gui_thread = threading.Thread(target=startGui, daemon=True)
        gui_thread.start()

    runGuiThread()

    while True:
        if key_states[keyboard.Key.up]:
            print('up')
        if key_states[keyboard.Key.down]:
            print('down')
        if key_states[keyboard.Key.left]:
            print('left')
        if key_states[keyboard.Key.right]:
            print('right')
        if key_states[keyboard.Key.alt_l]:
            print('alt_l')
        if key_states[keyboard.Key.alt_r]:
            print('alt_r')
        time.sleep(0.01)
    key_listener.join() 
