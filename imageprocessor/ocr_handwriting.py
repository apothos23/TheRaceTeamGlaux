# USAGE
# python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import time
from threading import Thread
from queue import Queue
import mouse
from PIL import Image, ImageDraw,ImageFont

def processimage():

    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)

    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames += "abcdefghijklmnopqrstuvwxyz"
    labelNames = [l for l in labelNames]

    str = ""

    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        str+=label
        
        # draw the prediction on the image
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # show the image
        cv2.imshow("image", image)
        #cv2.waitKey(0)
    print ("OCR " + str)    
    output = str
    
    
def getpredictions():

    # load the input image from disk, convert it to grayscale, and blur
    # it to reduce noise
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes, ensuring they are neither too small
        # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)

    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames += "abcdefghijklmnopqrstuvwxyz"
    labelNames = [l for l in labelNames]

    str = ""
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        str+=label
        
    print ("OCR " + str)    
    return str    
        
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600);

# 0 = raw 1 grayscale 2 processes 3 optimized
mode = 0

def updatephoto():
    #if cap.isOpened()
    for x in range(0, 10):
        #print("We're on time %d" % (x))    
        ret, frame = cap.read()
    #    if ret == False:
    #        break
    # Save Frame by Frame into disk using imwrite method
    global mode
    if mode == 0:
        cv2.imwrite('images/test.png', frame)
        cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)    
        cv2.imshow("image", frame)
        cv2.waitKey(0)

    if mode == 1:
        cv2.imwrite('images/test.png', frame)
        cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        frame = ~frame
        cv2.imshow("image", frame)
        cv2.waitKey(0)

    if mode == 2:
        cv2.imwrite('images/test.png', frame)
        cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)    
        cv2.imshow("image", frame)        
        processimage()
        
    if mode == 3:
        cv2.imwrite('images/test.png', frame)
        cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)            
        str = getpredictions()
        print ("OCR " + str)
        img = Image.new('RGB', (800, 600), color = (0, 0, 0))        
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype("Arial.ttf", 50)
        d.text((60,60), str, fill=(255,255,0) , font = font )        
        img.save('images/test.png')        
        #cv2.imwrite('images/test.png', frame)
        #cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)    
        cv2.imshow("image", img)
        cv2.waitKey(0)

def mouse_click(event, x, y, flags,param):
    
    # to check if left mouse 
    # button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:        
        # font for left click event
        print("left - update photo");
        updatephoto()
    if event == cv2.EVENT_MBUTTONDOWN:        
        # font for left click event
        global mode
        mode = (mode + 1) % 4;
        print("right new mode: " , mode);            


cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_click)

updatephoto()
