import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

class YOLOFaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection and Filters")

        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""

        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=10)

        btn_select_weights = tk.Button(btn_frame, text="Select Weights", command=self.select_weights)
        btn_select_weights.pack(side=tk.LEFT, padx=10)

        btn_select_cfg = tk.Button(btn_frame, text="Select CFG", command=self.select_cfg)
        btn_select_cfg.pack(side=tk.LEFT, padx=10)

        btn_select_names = tk.Button(btn_frame, text="Select Names", command=self.select_names)
        btn_select_names.pack(side=tk.LEFT, padx=10)

        btn_select_image = tk.Button(btn_frame, text="Select Image", command=self.select_image)
        btn_select_image.pack(side=tk.LEFT, padx=10)

        btn_select_video = tk.Button(btn_frame, text="Select Video", command=self.select_video)
        btn_select_video.pack(side=tk.LEFT, padx=10)

        btn_live_video = tk.Button(btn_frame, text="Live Video", command=self.toggle_live_video)
        btn_live_video.pack(side=tk.LEFT, padx=10)

        btn_detect_objects = tk.Button(btn_frame, text="Detect Objects", command=self.toggle_detect_objects)
        btn_detect_objects.pack(side=tk.LEFT, padx=10)

        btn_edge_detection = tk.Button(btn_frame, text="Edge Detection", command=self.toggle_edge_detection)
        btn_edge_detection.pack(side=tk.LEFT, padx=10)

        btn_sharpen = tk.Button(btn_frame, text="Sharpen", command=self.toggle_sharpen)
        btn_sharpen.pack(side=tk.LEFT, padx=10)

        self.image_path = None
        self.video_path = None
        self.image = None
        self.video_capture = None
        self.net = None
        self.classes = None
        self.output_layers = None
        self.filter_mode = None
        self.detect_objects_flag = False
        self.running = False
        self.thread = None

    def select_weights(self):
        self.weights_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_cfg(self):
        self.cfg_path = filedialog.askopenfilename()
        self.load_yolo()

    def select_names(self):
        self.names_path = filedialog.askopenfilename()
        self.load_yolo()

    def load_yolo(self):
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")

    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.load_image()

    def select_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            if self.running:
                self.stop_running()
            else:
                self.running = True
                self.thread = threading.Thread(target=self.detect_objects_video)
                self.thread.start()

    def toggle_live_video(self):
        if self.running:
            self.stop_running()
        else:
            self.video_capture = cv2.VideoCapture(0)
            self.running = True
            self.thread = threading.Thread(target=self.show_live_video)
            self.thread.start()

    def load_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.config(image=image)
        self.panel.image = image

    def toggle_detect_objects(self):
        self.detect_objects_flag = not self.detect_objects_flag
        if self.image_path:
            self.apply_filter_to_image()

    def toggle_edge_detection(self):
        if self.filter_mode == "edge":
            self.filter_mode = None
        else:
            self.filter_mode = "edge"
        if self.image_path:
            self.apply_filter_to_image()
        if not self.running:
            self.toggle_live_video()

    def toggle_sharpen(self):
        if self.filter_mode == "sharpen":
            self.filter_mode = None
        else:
            self.filter_mode = "sharpen"
        if self.image_path:
            self.apply_filter_to_image()
        if not self.running:
            self.toggle_live_video()

    def _detect_objects(self, frame):
        return self.apply_yolo(frame)

    def detect_objects_video(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.detect_objects_flag:
                frame = self.apply_yolo(frame)
            if self.filter_mode == "edge":
                frame = cv2.Canny(frame, 100, 200)
            elif self.filter_mode == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def show_live_video(self):
        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            if self.detect_objects_flag:
                frame = self.apply_yolo(frame)
            if self.filter_mode == "edge":
                frame = cv2.Canny(frame, 100, 200)
            elif self.filter_mode == "sharpen":
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow("Live Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop_running()

    def stop_running(self):
        self.running = False
        self.detect_objects_flag = False
        self.filter_mode = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        cv2.destroyAllWindows()

    def apply_yolo(self, image):
        if not self.net:
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")
            return image

        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if (x, y, w, h) and isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def apply_filter_to_image(self):
        image = cv2.imread(self.image_path)
        if self.detect_objects_flag:
            image = self.apply_yolo(image)
        if self.filter_mode == "edge":
            image = cv2.Canny(image, 100, 200)
        elif self.filter_mode == "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panel.config(image=image)
        self.panel.image = image

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOFaceDetectionApp(root)
    root.mainloop()
