import os
# from datetime import time
import time

from keras.models import load_model
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
my_model = models.Sequential()
# my_model = load_model("model/VGGTrained_model.h5")
my_model = load_model("model/VGGTrained_model3.h5")

# classes in our dataset

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# print(classes[y_class[0]])

# GUi Implementationn
root = tk.Tk()
root.geometry('1980x1080')
root.title('Sign Alphabet Detection')
root.configure(background='#CDCDCD')

label = Label(root, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(root, background='#CDCDCD')

# Sign prediction function
def classify(file_path):
    # global label_packed
    x = Image.open(file_path)
    x = x.resize((224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = my_model.predict(x)
    y_class = np.argmax(pred, axis=1)
    sign = classes[y_class[0]]
    label.configure(foreground='#40474C', text="Predicted Sign: " + sign)

    print(sign)


sentence = ""

#  Sentence prediction function
def classify_sentence(file_path):
    global sentence
    x = Image.open(file_path)
    x = x.resize((224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = my_model.predict(x)
    y_class = np.argmax(pred, axis=1)
    sign = classes[y_class[0]]

    if sign == 'del':
        sentence = sentence[:-1]
    elif sign == 'space':
        sentence = sentence + " "
    elif sign == 'nothing':
        sentence = sentence + " "
    else:
        sentence = sentence + sign
    label.configure(foreground='#40474C', text=sentence)

    print(sign)

#Upload image function
def upload_image():
    try:
        file_path = filedialog.askopenfilename(initialdir="/dataset/test/", title="Select image")
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width()), (root.winfo_height())))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)

    except:
        pass

# Show classify button function
def show_classify_button(file_path):
    classify_b = Button(root, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    # background = '#666464', foreground = 'white', font = ('arial', 10, 'bold'), borderwidth = 4, relief = "solid"
    classify_b.configure(background='#666464', foreground='white', font=('arial', 10, 'bold'), borderwidth=4,
                         relief="solid")
    classify_b.place(relx=0.79, rely=0.46)

# Click a picture from webcam function
def webcam_activated():
    print("Webcam activated, Press SPACE to take pic")

    cam = cv2.VideoCapture(0)
    window_name = "Sign Language Webcam"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cam.read()

        if not ret:
            print("failed to capture an image!")
            break
        # cam_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Sign", frame)

        # The webcam is on until a button is pressed
        k = cv2.waitKey(1)

        # k%256 == 27 means the escape key on your computer, if you hit it, the application closes
        if k % 256 == 27:
            print("Escape hit, closing the app")


            cam.release()
            cv2.destroyAllWindows()

            break

        # k%256 == 32 means the space button, if it is pressed, it runs the code below it to take a screenshot and save it in a png format.
        elif k % 256 == 32:
            img_name = "sign_frame.jpg"
            cv2.imwrite(img_name, frame)
            print("Screenshot taken")
            # cam_img = ImageTk.PhotoImage(Image.fromarray(cam_img))
            # cam_window['image'] = cam_img
            img_from_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_from_frame = ImageTk.PhotoImage(Image.fromarray(img_from_frame))
            sign_image.configure(image=img_from_frame)
            sign_image.image = img_from_frame
            root.update()
            classify(img_name)

        img_name = "sign_frame.jpg"
        cv2.imwrite(img_name, frame)
        # classify(img_name)
        root.update()



# Live detection function
def live_detection():
    print("izza webcam")

    cam = cv2.VideoCapture(0)
    window_name = "Sign Language Webcam"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cam.read()

        if not ret:
            print("failed to capture an image!")
            break

        cv2.imshow("Sign", frame)


        # The webcam is on until a button is pressed
        k = cv2.waitKey(100)

        # k%256 == 27 means the escape key on your computer, if you hit it, the application closes
        if k % 256 == 27:
            print("Escape hit, closing the app")

            label.configure(text="")

            cam.release()
            cv2.destroyAllWindows()
            break

        # k%256 == 32 means the space button, if it is pressed, it runs the code below it to take a screenshot and save it in a png format.
        elif k % 256 == 32:
            img_name = "sign_frame.jpg"
            cv2.imwrite(img_name, frame)
            print("Screenshot taken")
            # cam_img = ImageTk.PhotoImage(Image.fromarray(cam_img))
            # cam_window['image'] = cam_img
            img_from_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_from_frame = ImageTk.PhotoImage(Image.fromarray(img_from_frame))
            sign_image.configure(image=img_from_frame)
            sign_image.image = img_from_frame

            classify(img_name)

        img_name = "sign_frame.jpg"
        cv2.imwrite(img_name, frame)
        classify(img_name)

        img_from_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_from_frame = ImageTk.PhotoImage(Image.fromarray(img_from_frame))
        sign_image.configure(image=img_from_frame)
        sign_image.image = img_from_frame

        root.update()

# Sentence construction function
def sentence_construction():
    print("Sentence construction begings")
    global sentence

    cam = cv2.VideoCapture(0)
    window_name = "Sign Language Webcam"
    cv2.namedWindow(window_name)

    start_time = time.time()
    control_time = 5
    while True:
        ret, frame = cam.read()
        cv2.imshow("Sign", frame)
        k = cv2.waitKey(1)
        if not ret:
            print("failed to capture an image!")
            break

        if time.time() - start_time >= control_time:
            # display frame on label every two seconds
            img_from_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_from_frame = ImageTk.PhotoImage(Image.fromarray(img_from_frame))
            sign_image.configure(image=img_from_frame)
            sign_image.image = img_from_frame

            img_name = "sign_frame.jpg"
            cv2.imwrite(img_name, frame)
            classify_sentence(img_name)
            root.update()
            control_time = control_time+5

        # k%256 == 27 means the escape key on your computer, if you hit it, the application closes
        if k % 256 == 27:
            print("Escape hit, closing the app")

            cam.release()
            cv2.destroyAllWindows()
            sentence = ""
            break

        # root.update()

# All the button implementations and labels
button_frames = LabelFrame(root).pack()
webcam_button = Button(button_frames, text="Use Webcam", command=webcam_activated, padx=10, pady=5)
webcam_button.configure(background='#666464', foreground='white', font=('arial', 10, 'bold'), borderwidth=4,
                        relief="solid")
webcam_button.pack(side=BOTTOM, pady=50)

live_button = Button(button_frames, text="Live Detection", command=live_detection, padx=10, pady=5)
live_button.configure(background='#666464', foreground='white', font=('arial', 10, 'bold'), borderwidth=4,
                      relief="solid")
live_button.pack(side=BOTTOM, pady=50)

sentence_button = Button(button_frames, text="Sentence Construction", command=sentence_construction, padx=10, pady=5)
sentence_button.configure(background='#666464', foreground='white', font=('arial', 10, 'bold'), borderwidth=4,
                          relief="solid")
sentence_button.pack(side=BOTTOM, pady=50)

upload = Button(button_frames, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#666464', foreground='white', font=('arial', 10, 'bold'), borderwidth=4, relief="solid")
upload.pack(side=BOTTOM, pady=50)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(root, text="Sign Alphabet Detector", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#40474C')
heading.pack()

root.mainloop()
