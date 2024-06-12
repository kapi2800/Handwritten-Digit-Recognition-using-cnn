import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageGrab, ImageOps

# Load the trained model
model = tf.keras.models.load_model('modelcopy1.h5')

# Define the drawing class
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.canvas = Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_button = Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=2, padx=2)
        self.clear_button = Button(self.root, text="Clear", command=self.clear)
        self.clear_button.grid(row=1, column=1, pady=2, padx=2)

        self.prediction_label = Label(self.root, text="", font=("Helvetica", 24))
        self.prediction_label.grid(row=2, column=0, columnspan=2)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)

    def clear(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="")

    def predict(self):
        # Capture the canvas area and save it to an image file
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))

        # Preprocess the image
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        self.prediction_label.config(text=str(digit))

# Run the application
if __name__ == "__main__":
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()
