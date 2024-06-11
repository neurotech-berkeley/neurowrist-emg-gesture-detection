import cv2
import serial
import time
import subprocess
from threading import Thread
from datetime import datetime
import tkinter as tk
import random
import numpy as np

class DataRecorder:
    def __init__(self):
        self.fieldnames = ['millis', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'event']
        self.event_label = None
        self.event_labels = ["rest", "thumb-to-index", "thumb-to-pinky", "clench"]
        self.root = tk.Tk()
        self.root.title("Data Recorder")
        self.root.geometry("500x300")
        self.root.eval('tk::PlaceWindow . center')
        self.label = tk.Label(self.root, text="Initializing...", font=("Helvetica", 36))
        self.label.pack(pady=20, anchor="center")
        self.root.after(0, self.start_recording)
        self.window_closed = False

        self.data_dict = {}
        self.current_window = []
        self.data_ready = False

    def start_recording(self):
        self.label.configure(text="Rest", font=("Helvetica", 36))
        self.label.pack(pady=20, anchor="center")
        self.event_label = "rest"
        self.root.after(25000, self.flash_text)
        self.root.after(300000, self.stop_recording)

    def flash_text(self):
        label = self.event_label
        data = self.current_window
        sample_dict = {"label": label, "data": data}
        self.data_dict[len(self.data_dict)] = sample_dict
        self.current_window = []
        event_index = self.event_labels.index(self.event_label)

        if event_index < len(self.event_labels) - 1:
            self.label.configure(text=self.event_labels[event_index + 1], font=("Helvetica", 36))
            self.event_label = self.event_labels[event_index + 1]

        else:
            self.label.configure(text=self.event_labels[0], font=("Helvetica", 36))
            self.event_label = self.event_labels

        self.root.after(random.randint(1500, 3000), self.flash_text)

    def stop_recording(self):
        self.data_ready = True
        self.label.configure(text="Saving and closing...", font=("Helvetica", 16))
        self.label.pack(pady=20, anchor="center")
        self.root.after(2000, self.save_and_destroy)
        self.window_closed = True

    def save_and_destroy(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def select_uart_port():
    output = subprocess.run(["ls /dev/tty.*"], stdout=subprocess.PIPE, shell=True, text=True)
    ports = output.stdout.split("\n")[:-1]
    i = 0
    print("\n>> Select UART/USB to read from:\n")
    for port in ports:
        print("  ", i, ":", port)
        i += 1
    index = int(input("\n>> "))
    return ports[index]


port = select_uart_port()
print("\nOK, opening: ", port)


# Initialize serial connection
serialPort = serial.Serial(port=port, baudrate=115200, bytesize=8, timeout=0.1, stopbits=serial.STOPBITS_ONE)  # Adjust COM port and baud rate as necessary

data_recorder = DataRecorder()

# Initialize video recording
video_capture = cv2.VideoCapture(0)  # Adjust the index if using multiple cameras
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

data_stream = []

# Function to read data from the serial stream
def read_serial():
    while serialPort:
        x = serialPort.readline()
        try:
            millis, ch0, ch1, ch2, ch3, ch4 = x.decode("utf-8").split()[1:]
        except:
            continue
        timestamp = time.time()
        millis = int(millis)
        ch0 = int(ch0)
        ch1 = int(ch1)
        ch2 = int(ch2)
        ch3 = int(ch3)
        ch4 = int(ch4)
        row = [millis, ch0, ch1, ch2, ch3, ch4, data_recorder.event_label, timestamp]
        # print(millis, ch0, ch1, ch2, ch3, ch4, data_recorder.event_label, timestamp)
        data_stream.append(row)
        # # data_recorder.current_window.append(row[:-1])

        # if data_recorder.data_ready:
        #     now = datetime.now()
        #     dt_string = now.strftime("%m-%d-%Y-%H:%M:%S")
        #     save_as_npz = f"../data/unsorted/{dt_string}.npz"
        #     print(save_as_npz)
            
        #     # Convert keys to strings before saving
        #     data_dict_str_keys = {str(key): value for key, value in data_recorder.data_dict.items()}
            
        #     np.savez(save_as_npz, **data_dict_str_keys)
        #     break

        # if not data_recorder.window_closed:
        #     data_recorder.root.update_idletasks()
        #     data_recorder.root.update()
        # else:
        #     break

    # while True:
    #     if ser.in_waiting > 0:
    #         line = ser.readline().decode('utf-8').strip()
    #         timestamp = time.time()
    #         data_stream.append((timestamp, line))


# Start the serial reading thread
serial_thread = Thread(target=read_serial)
serial_thread.start()

try:
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            timestamp = time.time()
            # Write the frame with timestamp
            out.write(frame)
            # Optionally, display the frame
            cv2.imshow('frame', frame)

            # Press 'q' to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
except KeyboardInterrupt:
    pass

# Release everything
video_capture.release()
out.release()
cv2.destroyAllWindows()

# Save the data stream to a file for later synchronization
with open('data_stream.txt', 'w') as f:
    for timestamp, data in data_stream:
        f.write(f"{timestamp},{data}\n")