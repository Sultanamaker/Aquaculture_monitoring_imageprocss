from flask import Flask, render_template, request, redirect, url_for, session, Response
import re
from numpy.linalg import norm
import cv2
import sys
import os
import math
import numpy as np
import json
from threading import Thread
import VideoEnhancement
import fishpredictor
import detector
import kmeancluster
import preproccesing
import randomforst
import operator

class fishs:
    def __init__(self):
        self.mylist = []
    def addfish(self,data):
        x=[data]
        self.mylist.append(x)
    def addframe(self,id,data):
        # print(len(self.mylist))
        if len(self.mylist)>(id-1):
            self.mylist[id-1].append(data)
        else:
            self.addfish(data)


app = Flask(__name__)
app.secret_key = "abc"

def Generator():
    try :
        waterIsToxic = "Clear"
        isFinished = False
        currentBehavior = "Normal"
        cap = cv2.VideoCapture("http://192.168.68.112:81/stream")
        cluster = kmeancluster.kmeans()
        classifier = randomforst.randomforst()
        fishs = []
        framenum = 0
        sum = 0
        max = 0
        mylist = [[]]
        yolo = detector.detector()
        ret, frame = cap.read()
        frame = frame
        fheight, fwidth, channels = frame.shape
        resize = False
        if (fheight > 352 or fwidth > 640):
            resize = True
            fwidth = 640
            fheight = 352
            frame = cv2.resize(frame, (640, 352))

        mask = np.zeros_like(frame)
        # Needed for saving video
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_seconds = 10
        # Read until video is completed
        buffer = [[]]
        apperance = [[]]
        last_changed = []
        top = 0
        frms = 0

        # Needed to track objects
        n_frame = 8
        ref_n_frame_axies = []
        ref_n_frame_label = []
        ref_n_frame_axies_flatten = []
        ref_n_frame_label_flatten = []
        frm_num = 1
        coloredLine = np.random.randint(0, 255, (10000, 3))
        label_cnt = 1
        min_distance = 50
        while (cap.isOpened()):
            ret, img = cap.read()
            img = img
            if ret == True:
                if frms % 2 == 0:
                    img = VideoEnhancement.enhanceVideo(img, resize)
                    cur_frame_axies = []
                    cur_frame_label = []
                    boxes, confidences, centers, colors = yolo.detect(img)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
                    font = cv2.FONT_HERSHEY_PLAIN
                    for i in range(len(boxes)):
                        if i in indexes:
                            lbl = float('nan')
                            x, y, w, h, = boxes[i]
                            center_x, center_y = centers[i]
                            color = colors[0]
                            if (len(ref_n_frame_label_flatten) > 0):
                                b = np.array([(center_x, center_y)])
                                a = np.array(ref_n_frame_axies_flatten)
                                distance = norm(a - b, axis=1)
                                min_value = distance.min()
                                if (min_value < min_distance):
                                    idx = np.where(distance == min_value)[0][0]
                                    lbl = ref_n_frame_label_flatten[idx]
                                    points = (int(ref_n_frame_axies_flatten[idx][0]), int(ref_n_frame_axies_flatten[idx][1]))
                                    mask = cv2.line(mask, (center_x, center_y), points, coloredLine[lbl].tolist(), 2)
                                    cv2.circle(img, points, 5, coloredLine[lbl].tolist(), -1)

                            if (math.isnan(lbl)):
                                lbl = label_cnt
                                label_cnt += 1

                            cur_frame_label.append(lbl)
                            cur_frame_axies.append((center_x, center_y))

                            fishs.append([lbl, x, y, w, h])
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img, '{}{}'.format("Fish", lbl), (x, y - 5), font, 1, (255, 255, 255), 2)

                    if (len(ref_n_frame_axies) == n_frame):
                        del ref_n_frame_axies[0]
                        del ref_n_frame_label[0]

                    ref_n_frame_label.append(cur_frame_label)
                    ref_n_frame_axies.append(cur_frame_axies)
                    ref_n_frame_axies_flatten = [a for ref_n_frame_axie in ref_n_frame_axies for a in ref_n_frame_axie]
                    ref_n_frame_label_flatten = [b for ref_n_frame_lbl in ref_n_frame_label for b in ref_n_frame_lbl]
                    sortedfish = sorted(fishs, key=operator.itemgetter(0))
                    fishs = []

                    if (len(sortedfish) != 0):
                        fishpredictor.predictfish(sortedfish, apperance, buffer, last_changed, top, img, color, mylist, framenum)

                    img = cv2.add(img, mask)
                    # cv2.imshow("Image", img)
                    mylist.append([])
                    framenum += 1
                    print(frms)
                    print("----------")
                    # cap.set(1,frms)
                    if (frms % (round(fps) * num_seconds) == 0 and frms!=0):
                        result = cluster.classify(mask)

                        randomForestResult = (classifier.classify(sortedfish, mask, fps))
                        print(randomForestResult)


                        print("result " + str(result))
                        if (randomForestResult[0] == 1):
                            currentBehavior = "Hunger"
                        
                        elif(randomForestResult[0] == 0):
                            currentBehavior = "Normal"
                        
                        elif(randomForestResult[0] == 2):
                            currentBehavior = "Obstacle in pond"
                            
                        mask = np.zeros_like(frame)
                        ref_n_frame_axies = []
                        ref_n_frame_label = []
                        ref_n_frame_axies_flatten = []
                        ref_n_frame_label_flatten = []
                        buffer = [[]]
                        apperance = [[]]
                        last_changed = []
                        # frms = 0
                        mylist = [[]]
                        framenum = 0

                        label_cnt = 1
                        top = 0
                frms += 1
                _, buffer = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except: 
        print("")
@app.route('/video')
def video():
    return Response(Generator(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=True)
