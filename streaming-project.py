#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date created: May 5th, 2020

Project:
Multiple Pedestrian Tracking (MPT) Over Distributed Camera Network (DNS)

@author: Lixian Liu
"""
import time
import cv2
from flask import Flask, render_template, Response, request, session
import glob
import numpy
import os
from multiprocessing import Process


app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def index():
    return render_template('index.html')

#get number of frames from HTML Form
@app.route('/', methods=['POST'])
def value():
    frame_num = request.form['frame_num']
    session['f1'] = frame_num
    return render_template('index.html')

#########################################
'''video address records'''
###live webcam; desktop###
'''http://99.229.86.67:8081/video.mjpg'''   

###local file; original gen2()###
'''/home/guitracker/multicamera/PETS09S2L1/View_005.avi'''

###local file; original gen3()###
'''/home/guitracker/multicamera/PETS09S2L1/View_001.avi'''

###live camera; spring pool###
'''http://208.72.70.171:80/mjpg/video.mjpg'''

###live camera; golf club###
'''http://wmccpinetop.axiscam.net/mjpg/video.mjpg'''

###live webcam; old labtop###
'''http://99.229.86.67:8080/video.mjpg'''

###multiple videos###
'''terrace1-c0,c1,c2,c3.avi'''
#########################################



"""
#training for the live video from webcam
def train_gen():   
    f_num = session.get('f1', None)
    
    if(os.path.exists('../output2/outputtest')):
        None
    else:    
        if (f_num != None):
            os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video http://99.229.86.67:8081/video.mjpg --output_dir /home/guitracker/Summer-Research/output2 --frame_count ' + f_num)
"""   
    
#local video inference
def gen():
    """
    #camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture('http://99.229.86.67:8081/video.mjpg')
    
    #camera = cv2.VideoCapture('http://208.72.70.171:80/mjpg/video.mjpg')
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
    """
    while True:
        if os.path.isfile("../output_local_inference/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        
    
    i = 0
    while True:
        if os.path.isfile("../output_local_inference/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_local_inference/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1
        else:
            cap = cv2.imread("../output_local_inference/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
#end of local video inference

#local video inference track
def gentrk():

    while True:
        if os.path.isfile("../output_local_inference_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
    i = 0
    while True:
        if os.path.isfile("../output_local_inference_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_local_inference_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1
        else:
            cap = cv2.imread("../output_local_inference_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
#end of local video inference track

"""
#streaming the video from Tiger; under PETS09S2L1/View_005.avi
def gen():
    #cap = cv2.VideoCapture('video.mp4') #original local file
    
    cap = cv2.VideoCapture('../../multicamera/PETS09S2L1/View_005.avi') #testing the file from the Tiger

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break
"""
#training function for 2nd video button
def train_gen2():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video /home/guitracker/multicamera/PETS09S2L1/View_005.avi --output_dir /home/guitracker/Summer-Research/output_local_live --frame_count ' + f_num)
    
#function for streaming a single picture
def gen2():
    #generate video streaming
    #img = cv2.imread("yorku.jpg") #reading single image
    #img = [cv2.imread(file) for file in glob.glob("../output1/outputtest/*.jpg")]
    """
    cap = cv2.VideoCapture("../output123/outputtest/frame_%0d.jpg", cv2.CAP_IMAGES)
    while True:
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
        else:
            break
    """
    """
    #reading the image in sequence instead of using VideoCapture
    i = 0
    while True:
        cap = cv2.imread("../output123/outputtest/frame_" + str(i) + ".jpg")
        cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
        frame = cv2.imencode('.jpg', cap)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.15)
        i = i + 1
    """
    
    while True:
        if os.path.isfile("../output_local_live/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_local_live/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_local_live/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_local_live/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            
#training tracking function for local video (real time)
def train_gen2trk():   
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_local_live/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_local_live/outputtest --output_dir /home/guitracker/Summer-Research/output_local_live_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
        
#function for local video live training (tracking)
def gen2trk():

    while True:
        if os.path.isfile("../output_local_live_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_local_live_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_local_live_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_local_live_trk/camera1/' + str("%06d" % i) + '.jpg ../output_local_live_trk/camera2/' + str("%06d" % i) + '.jpg' )
            cap = cv2.imread("../output_local_live_trk/camera2/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            
"""
#function for streaming video from local file
def gen2():
    #cap = cv2.VideoCapture('video.mp4') #original local file
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../../multicamera/PETS09S2L1/View_001.avi') #testing the file from the Tiger

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1) #for streaming video from Tiger
            #time.sleep(0.02) #for streaming video from local computer
        else:
            break
 """
 
#train the 3rd video if the pre-trained folder does not exist
def train_gen3():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video http://99.229.86.67:8081/video.mjpg --output_dir /home/guitracker/Summer-Research/output_webcam --frame_count ' + f_num)
    
    
#function for streaming the 3rd video
def gen3():
    #cap = cv2.VideoCapture('video.mp4') #original local file
    
    #cap = cv2.VideoCapture('../../multicamera/PETS09S2L1/View_006.avi') #testing the file from the Tiger
    
    """
    cap = cv2.VideoCapture("../output1/outputtest/frame_%0d.jpg", cv2.CAP_IMAGES) #test for detection output1

    #while(cap.isOpened()):
    while True:
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.45)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1) #for streaming video from Tiger
            #time.sleep(0.02) #for streaming video from local computer
        else:
            break
    """
    """
    #for reading images in sequence
    for count in range(13):
        cap = cv2.imread("../output1/outputtest/frame_" + str(count) + ".jpg") 
        cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
        frame = cv2.imencode('.jpg', cap)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.2)
    """
    #while (not os.path.isfile("../output1/outputtest/0.txt")):
    
    
    while True:
        if os.path.isfile("../output_webcam/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_webcam/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_webcam/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_webcam/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
    
#train the 3rd video if the pre-trained folder does not exist (tracking)
def train_gen3trk():   
    
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_webcam/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_webcam/outputtest --output_dir /home/guitracker/Summer-Research/output_webcam_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
        
    
#function for streaming the 3rd video (tracking)
def gen3trk():
    
    
    while True:
        if os.path.isfile("../output_webcam_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_webcam_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_webcam_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_webcam_trk/camera1/' + str("%06d" % i) + '.jpg ../output_webcam_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_webcam_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            
#train the golf video if the pre-trained folder does not exist
def train_gen4():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video http://wmccpinetop.axiscam.net/mjpg/video.mjpg --output_dir /home/guitracker/Summer-Research/output_live_golf --frame_count ' + f_num)
       
#streaming and training for golf club video
def gen4():
    """
    cap = cv2.VideoCapture('http://wmccpinetop.axiscam.net/mjpg/video.mjpg') 

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1) #for streaming video from Tiger
            #time.sleep(0.02) #for streaming video from local computer
        else:
            break
    """
    while True:
        if os.path.isfile("../output_live_golf/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            #time.sleep(1)
    
    i = 0
    while True:
        if os.path.isfile("../output_live_golf/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_live_golf/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_live_golf/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)

#train the golf video if the pre-trained folder does not exist(tracking)
def train_gen4trk():   
    
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_live_golf/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_live_golf/outputtest --output_dir /home/guitracker/Summer-Research/output_live_golf_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break   
    
#streaming and training for golf club video(tracking)
def gen4trk():

    while True:
        if os.path.isfile("../output_live_golf_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_live_golf_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_live_golf_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_live_golf_trk/camera1/' + str("%06d" % i) + '.jpg ../output_live_golf_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_live_golf_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            
####################################################
'''multiple videos section gen5 to gen8'''
####################################################

#training for terrace-c0
def train_gen5():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video /home/guitracker/multicamera/EPFL/Terrace/terrace1-c0.avi --output_dir /home/guitracker/Summer-Research/output_terrace_c0 --frame_count ' + f_num)
 
#for terrace-c0
def gen5():
    """
    cap = cv2.VideoCapture('http://wmccpinetop.axiscam.net/mjpg/video.mjpg') 

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1) #for streaming video from Tiger
            #time.sleep(0.02) #for streaming video from local computer
        else:
            break      
    """
    while True:
        if os.path.isfile("../output_terrace_c0/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c0/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_terrace_c0/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_terrace_c0/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)

#training for terrace-c0 (tracking)
def train_gen5trk():   
  
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_terrace_c0/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_terrace_c0/outputtest --output_dir /home/guitracker/Summer-Research/output_terrace_c0_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
    
#for terrace-c0 (tracking)
def gen5trk():
            
    while True:
        if os.path.isfile("../output_terrace_c0_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c0_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_terrace_c0_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_terrace_c0_trk/camera1/' + str("%06d" % i) + '.jpg ../output_terrace_c0_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_terrace_c0_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)   
                 
#training for terrace-c1
def train_gen6():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video /home/guitracker/multicamera/EPFL/Terrace/terrace1-c1.avi --output_dir /home/guitracker/Summer-Research/output_terrace_c1 --frame_count ' + f_num)
              
#for terrace-c1
def gen6():
    
    while True:
        if os.path.isfile("../output_terrace_c1/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c1/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_terrace_c1/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_terrace_c1/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)

#training for terrace-c1 (tracking)
def train_gen6trk():   
  
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_terrace_c1/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_terrace_c1/outputtest --output_dir /home/guitracker/Summer-Research/output_terrace_c1_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
    
#for terrace-c1 (tracking)
def gen6trk():
            
    while True:
        if os.path.isfile("../output_terrace_c1_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c1_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_terrace_c1_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_terrace_c1_trk/camera1/' + str("%06d" % i) + '.jpg ../output_terrace_c1_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_terrace_c1_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)   
            
#training for terrace-c2
def train_gen7():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video /home/guitracker/multicamera/EPFL/Terrace/terrace1-c2.avi --output_dir /home/guitracker/Summer-Research/output_terrace_c2 --frame_count ' + f_num)
    
#for terrace-c2
def gen7():
    
    while True:
        if os.path.isfile("../output_terrace_c2/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c2/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_terrace_c2/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_terrace_c2/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)           

#training for terrace-c2 (tracking)
def train_gen7trk():   
  
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_terrace_c2/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_terrace_c2/outputtest --output_dir /home/guitracker/Summer-Research/output_terrace_c2_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
    
#for terrace-c2 (tracking)
def gen7trk():
            
    while True:
        if os.path.isfile("../output_terrace_c2_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c2_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_terrace_c2_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_terrace_c2_trk/camera1/' + str("%06d" % i) + '.jpg ../output_terrace_c2_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_terrace_c2_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)   
 
#training for terrace-c3
def train_gen8():   
    f_num = session.get('f1', None)
    if (f_num != None):
        os.system('python /home/guitracker/maskrcnn/maskrcnnv1/maininference/main2.py --display --input_video /home/guitracker/multicamera/EPFL/Terrace/terrace1-c3.avi --output_dir /home/guitracker/Summer-Research/output_terrace_c3 --frame_count ' + f_num)
    
#for terrace-c3
def gen8():
    
    while True:
        if os.path.isfile("../output_terrace_c3/outputtest/frame_0.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c3/outputtest/frame_" + str(i + 1) + ".jpg"):
            cap = cv2.imread("../output_terrace_c3/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            cap = cv2.imread("../output_terrace_c3/outputtest/frame_" + str(i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            
#training for terrace-c3 (tracking)
def train_gen8trk():   
  
    f_num = session.get('f1', None)
    j = int(int(f_num) / 4)
    ct = j
    while True:
        if os.path.isfile("../output_terrace_c3/outputtest/frame_"+ str(ct) + ".jpg"):
            os.system('python /home/guitracker/mpt/maintrkcam2.py --display --n camera1 --waitingtime 10000 --input_detectfile /home/guitracker/Summer-Research/output_terrace_c3/outputtest --output_dir /home/guitracker/Summer-Research/output_terrace_c3_trk/ --frame_count ' + str(f_num))
            ct = int(ct) + j
            if(ct >= int(f_num)):
                break
    
#for terrace-c3 (tracking)
def gen8trk():
            
    while True:
        if os.path.isfile("../output_terrace_c3_trk/camera1/000000.jpg"):
            break
        else:
            cap = cv2.imread("./loading.jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    
    i = 0
    while True:
        if os.path.isfile("../output_terrace_c3_trk/camera1/" + str("%06d" % (i + 1)) + ".jpg"):
            cap = cv2.imread("../output_terrace_c3_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)
            i = i + 1

        else:
            os.system('cp ../output_terrace_c3_trk/camera1/' + str("%06d" % i) + '.jpg ../output_terrace_c3_trk/camera2/' + str("%06d" % i) + '.jpg' )

            cap = cv2.imread("../output_terrace_c3_trk/camera1/" + str("%06d" % i) + ".jpg")
            cap = cv2.resize(cap, (0,0), fx = 0.5, fy = 0.5)
            frame = cv2.imencode('.jpg', cap)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.15)   
            
#read a local video inference
@app.route('/local_video_inference')
def local_video_inference():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#read from local inference track
@app.route('/local_video_inference_trk')
def local_video_inference_trk():
    return Response(gentrk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#training a local video file
@app.route('/local_video_live')
def local_video_live():
    if(os.path.exists('../output_local_live/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_local_live/outputtest') ###
    global g2
    g2 = Process(target = train_gen2)
    g2.start()
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#training tracking component of a local video file
@app.route('/local_video_live_trk')
def local_video_live_trk():
    if(os.path.exists('../output_local_live_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../output_local_live_trk/camera1') ###
        os.system('rm -rf ../output_local_live_trk/camera1.txt')
        os.system('rm -rf ../output_local_live_trk/camera2')
        os.system('mkdir ../output_local_live_trk/camera2')
    global g2trk
    g2trk = Process(target = train_gen2trk)
    g2trk.start()
    return Response(gen2trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#streaming a webcam from desktop
@app.route('/video_webcam')
def video_webcam():
    if(os.path.exists('../output_webcam/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_webcam/outputtest') ###
    global g3
    g3 = Process(target = train_gen3)
    g3.start()
    return Response(gen3(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#streaming a webcam from desktop (tracking)
@app.route('/video_webcam_trk')
def video_webcam_trk():
    if(os.path.exists('../output_webcam_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../output_webcam_trk/camera1') ###
        os.system('rm -rf ../output_webcam_trk/camera1.txt')
        os.system('rm -rf ../output_webcam_trk/camera2')
        os.system('mkdir ../output_webcam_trk/camera2')
    global g3trk
    g3trk = Process(target = train_gen3trk)
    g3trk.start()
    return Response(gen3trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
            
#training a live golf video
@app.route('/video_live_golf')
def video_live_golf():
    if(os.path.exists('../output_live_golf/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_live_golf/outputtest') ###
    global g4
    g4 = Process(target = train_gen4)
    g4.start()
    return Response(gen4(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#training a live golf video (tracking)
@app.route('/video_live_golf_trk')
def video_live_golf_trk():
    if(os.path.exists('../output_live_golf_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../output_live_golf_trk/camera1') ###
        os.system('rm -rf ../output_live_golf_trk/camera1.txt')
        os.system('rm -rf ../output_live_golf_trk/camera2')
        os.system('mkdir ../output_live_golf_trk/camera2')
    global g4trk
    g4trk = Process(target = train_gen4trk)
    g4trk.start()
    return Response(gen4trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')    
############################################
    '''multiple videos section'''
############################################
#terrace-c0
@app.route('/video_terrace_c0')
def video_terrace_c0():
    if(os.path.exists('../output_terrace_c0/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_terrace_c0/outputtest') ###
    global g5
    g5 = Process(target = train_gen5)
    g5.start()
    
    return Response(gen5(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#terrace-c0 (tracking)
@app.route('/video_terrace_c0_trk')
def video_terrace_c0_trk():
    if(os.path.exists('../video_terrace_c0_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../video_terrace_c0_trk/camera1') ###
        os.system('rm -rf ../video_terrace_c0_trk/camera1.txt')
        os.system('rm -rf ../video_terrace_c0_trk/camera2')
        os.system('mkdir ../video_terrace_c0_trk/camera2')
    global g5trk
    g5trk = Process(target = train_gen5trk)
    g5trk.start()
    return Response(gen5trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  
    
#terrace-c1
@app.route('/video_terrace_c1')
def video_terrace_c1():
    if(os.path.exists('../output_terrace_c1/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_terrace_c1/outputtest') ###
    global g6
    g6 = Process(target = train_gen6)
    g6.start()
    return Response(gen6(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#terrace-c1 (tracking)
@app.route('/video_terrace_c1_trk')
def video_terrace_c1_trk():
    if(os.path.exists('../video_terrace_c1_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../video_terrace_c1_trk/camera1') ###
        os.system('rm -rf ../video_terrace_c1_trk/camera1.txt')
        os.system('rm -rf ../video_terrace_c1_trk/camera2')
        os.system('mkdir ../video_terrace_c1_trk/camera2')
    global g6trk
    g6trk = Process(target = train_gen6trk)
    g6trk.start()
    return Response(gen6trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

#terrace-c2
@app.route('/video_terrace_c2')
def video_terrace_c2():
    if(os.path.exists('../output_terrace_c2/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_terrace_c2/outputtest') ###
    global g7
    g7 = Process(target = train_gen7)
    g7.start()
    return Response(gen7(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#terrace-c2 (tracking)
@app.route('/video_terrace_c2_trk')
def video_terrace_c2_trk():
    if(os.path.exists('../video_terrace_c2_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../video_terrace_c2_trk/camera1') ###
        os.system('rm -rf ../video_terrace_c2_trk/camera1.txt')
        os.system('rm -rf ../video_terrace_c2_trk/camera2')
        os.system('mkdir ../video_terrace_c2_trk/camera2')
    global g7trk
    g7trk = Process(target = train_gen7trk)
    g7trk.start()
    return Response(gen7trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

#terrace-c3
@app.route('/video_terrace_c3')
def video_terrace_c3():
    if(os.path.exists('../output_terrace_c3/outputtest')): ### remove the folder first
        os.system('rm -rf ../output_terrace_c3/outputtest') ###
    global g8
    g8 = Process(target = train_gen8)
    g8.start()
    return Response(gen8(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
#terrace-c3 (tracking)
@app.route('/video_terrace_c3_trk')
def video_terrace_c3_trk():
    if(os.path.exists('../video_terrace_c3_trk/camera1')): ### remove the folder first
        os.system('rm -rf ../video_terrace_c3_trk/camera1') ###
        os.system('rm -rf ../video_terrace_c3_trk/camera1.txt')
        os.system('rm -rf ../video_terrace_c3_trk/camera2')
        os.system('mkdir ../video_terrace_c3_trk/camera2')
    global g8trk
    g8trk = Process(target = train_gen8trk)
    g8trk.start()
    return Response(gen8trk(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')  

   
if __name__=='__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 5000, threaded = True)
    


