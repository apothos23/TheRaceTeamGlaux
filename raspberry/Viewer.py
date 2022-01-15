import cv2
import zmq
import base64
import numpy as np
import screeninfo

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:5555')
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

while True:
    try:
        frame = footage_socket.recv_string()
        img = base64.b64decode(frame)
        npimg = np.fromstring(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        #exei keno sthn o8onh
        cv2.namedWindow("window",cv2.WND_PROP_FULLSCREEN)
        cv2.namedWindow("window",1)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        cv2.imshow("window", gray)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break
