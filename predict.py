# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2
import random as rd
import tensorflow as tf
import requests
import time
import base64
import json
import keras
import queue 
import threading, time, random
import requests
import ast

from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnnV1 import config
from mrcnnV1 import utils
import mrcnnV1.model as modellib
import datetime
import training_multi
import variable as VAR

def LoadingModel(model_dir,model_type,q_model):
    '''
# loading MASK R-CNN model
# determine parameter of model which is morning or night
    '''
    
    # Reset and create tensorflow graph
    tf.reset_default_graph()
    sess = tf.Session()
    graph = tf.get_default_graph()
    keras.backend.set_session(sess)
    
    # Change class depend on model type
    if model_type == "morning":
        num_classes =  len(VAR.class_names_morning)
    else:
        num_classes =  len(VAR.class_names_night)
    
    # Model config information
    class InferenceConfig(training_multi.SampleConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES =  num_classes
        BACKBONE = "resnet101"
        DETECTION_MIN_CONFIDENCE = 0.8
        #MINI_MASK_SHAPE = (56, 56)    
    config = InferenceConfig()
    config.display()
     
    # Loading config to model and load model weight
    model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config) # Create model object in inference mode.
    model.load_weights(model_dir, by_name=True)
    model.keras_model._make_predict_function()
    
    # Put model in queue
    q_model.put(model)

def CheckParamExist(tag,params):
    '''
# check parameter from post is exist or not
    '''
    value = None
    if tag.upper() in params:
        value = params[tag.upper()]
    return value

def Line_Notify(image_dir, true_time, ccd_uuid,name): # sending Line notify
    '''
# If event exists, use Line Notify
    '''
    
    try:
        message = ""
        for i in range(len(name)):
            message += 'https://cam.focusit.com.tw/AIR/UpdateResultAnalysis?uuid=' + ccd_uuid + '&date=' + true_time.strftime("%Y-%m-%d %H:%M:%S") + '&name=' + name[i].split(",")[2] + '&token= \n'
        img = open(image_dir,mode='rb')
        data = {'message': message}
        file = {"imageFile":img}
    
        response = requests.post(VAR.LINEADDRESS,headers={'Authorization': "Bearer "+VAR.LINE_TOKEN},data=data,files=file)
        
    except Exception as e:
        
        print(e)

def cv2ImgAddText(img,text,position,textColor=(0, 0, 0), textSize=20):
    '''
# Opencv can't present chinese text, use PIL instead
    '''
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    fontText = ImageFont.truetype("NotoSansTC-Regular.otf", textSize, encoding="utf-8")

    draw.text(position, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 將RGB影像轉換為Base64格式
def Image2Base64(image):
    with BytesIO() as buffer:
        resized_image_byte = Image.fromarray(image, "RGB") # array to image format
        resized_image_byte.save(buffer,format="JPEG") # store image in byte buffer
        imagebase64 = base64.b64encode(buffer.getvalue()) # turn byte value into base64 format
    return imagebase64

# 更換白天/夜間模型
def ChangeModel(model_dir,model_type):
    # 建立一個佇立
    queue_model = queue.Queue()
    # 建立一個執行緒，並且把佇列傳遞給執行緒
    thread_model = threading.Thread(target=LoadingModel,args=(model_dir,model_type,queue_model,))
    # 開始執行緒
    thread_model.start()
    # 阻塞主執行緒，等待執行緒(thread_model)結束
    thread_model.join()
    model = queue_model.get()    
    return model

# 寫入模型類型
def WriteModelType(model_type):
    with open("model_type.txt","w") as f:
        f.write(model_type)

# 影像預處理
def ImagePreprocessing(undo_img):
    # 在影像辨識前先針對不同的攝像頭影像作預處理，目前只採用資料增陽
    def his(img):
        # 夜晚模型採用直方圖均衡CLAHE增強
        if model_type == "night":
            ycrcb = cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
            channels = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(2,2))
            clahe.apply(channels[0],channels[0])
            cv2.merge(channels,ycrcb)
            cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2RGB,img)
        # 早晨模型採用淡化顏色
        else:
            # 創建一個灰色影像
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # 淡化顏色
            img = cv2.addWeighted(img, 0.5, gray_image, 1 - 0.5, 0)
        return img
    
    done_image = cv2.resize(undo_img,VAR.IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
    done_image =his(done_image)

    return done_image

# 儲存第一次事件的影像
def SaveFirstEventImage(image,ccd_uuid,true_time):
    now_time = true_time.strftime("%Y%m%d%H%M%S")
    today_time = true_time.strftime("%Y%m%d")
    store_path = os.path.join("first_event",ccd_uuid,today_time)
    os.makedirs(store_path,exist_ok=True)
    cv2.imwrite(os.path.join(store_path,now_time+"_1.jpg"),image)

# 取得太陽升起和落下時間
def GetSundata(true_time):
    # get sun data from government
    sun_data_json = dict()
    if os.path.exists("sun_data.json"):
        sun_data_json = json.load(open("sun_data.json"))
        if true_time.strftime("%Y%m%d") not in list(sun_data_json.keys()):
            today_sun_data = json.loads(requests.get(VAR.GOV_DATA_SUNRISE_SUNSET_URL,params={"CountyName":"高雄市","Date":true_time.strftime("%Y-%m-%d")}).text)
            sun_data_json[true_time.strftime("%Y%m%d")] = {"SunRiseTime":today_sun_data['records']['locations']['location'][0]['time'][0]["SunRiseTime"],"SunSetTime":today_sun_data['records']['locations']['location'][0]['time'][0]["SunSetTime"]}
            with open("sun_data.json","w") as f:
                json.dump(sun_data_json,f)

        sunrise_time = sun_data_json[true_time.strftime("%Y%m%d")]["SunRiseTime"]
        sunset_time = sun_data_json[true_time.strftime("%Y%m%d")]["SunSetTime"]
    
    else:
        today_sun_data = json.loads(requests.get(VAR.GOV_DATA_SUNRISE_SUNSET_URL,params={"CountyName":"高雄市","Date":true_time.strftime("%Y-%m-%d")}).text)
        sun_data_json[true_time.strftime("%Y%m%d")] = {"SunRiseTime":today_sun_data['records']['locations']['location'][0]['time'][0]["SunRiseTime"],"SunSetTime":today_sun_data['records']['locations']['location'][0]['time'][0]["SunSetTime"]}
        
        sunrise_time = today_sun_data['records']['locations']['location'][0]['time'][0]["SunRiseTime"]
        sunset_time = today_sun_data['records']['locations']['location'][0]['time'][0]["SunSetTime"]
        
        with open("sun_data.json","w") as f:
            json.dump(sun_data_json,f)
    return sunrise_time,sunset_time 

# 繪製監測項目邊界框
def DrawBoundingBox(ccd_tag,origin_shape,img):
    # 繪製監測項目邊界框
    new_x_coordinate1 = int(ccd_tag["START"][0]/(origin_shape[1]/1024) - (ccd_tag["END"][0]/(origin_shape[1]/1024) - ccd_tag["START"][0]/(origin_shape[1]/1024))*0.15) #resize rectangle size
    new_y_coordinate1 = int(ccd_tag["START"][1]/(origin_shape[0]/576) - (ccd_tag["END"][1]/(origin_shape[0]/576) - ccd_tag["START"][1]/(origin_shape[0]/576)) * 0.25)
    new_x_coordinate2 = int(ccd_tag["END"][0]/(origin_shape[1]/1024) + (ccd_tag["END"][0]/(origin_shape[1]/1024) - ccd_tag["START"][0]/(origin_shape[1]/1024))*0.15)
    new_y_coordinate2 = int(ccd_tag["END"][1]/(origin_shape[0]/576))              
    if new_y_coordinate1 <= 0:
        new_y_coordinate1 = 10
    for idx_x in range(new_x_coordinate2 - new_x_coordinate1):
        if idx_x %3 == 0:
            img = cv2.circle(img,(new_x_coordinate1+idx_x,new_y_coordinate1),1,(0,255,0),-1)
            img = cv2.circle(img,(new_x_coordinate1+idx_x,new_y_coordinate2),1,(0,255,0),-1)
    for idx_y in range(new_y_coordinate2 - new_y_coordinate1):
        if idx_y %3 == 0:
            img = cv2.circle(img,(new_x_coordinate1,new_y_coordinate1+idx_y),1,(0,255,0),-1)
            img = cv2.circle(img,(new_x_coordinate2,new_y_coordinate1+idx_y),1,(0,255,0),-1) 
    return new_x_coordinate1,new_y_coordinate1,new_x_coordinate2,new_y_coordinate2,img

#### 條件比對 ####

# 黑煙顏色比對
def compareColor(mask,img):
    # 顏色比對
    mask_r = np.array(np.where(mask > 0,img[:,:,0].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_g = np.array(np.where(mask > 0,img[:,:,1].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_b = np.array(np.where(mask > 0,img[:,:,2].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_r = mask_r[mask_r != None]
    mask_g = mask_g[mask_g != None]
    mask_b = mask_b[mask_b != None]
    
    count = 0

    bgr_value = np.array(np.dstack((mask_b,mask_g,mask_r)),dtype='uint8')
    hsv_lower = cv2.cvtColor(bgr_value,cv2.COLOR_BGR2HSV)
    h = hsv_lower[:,:,0].flatten()
    s = hsv_lower[:,:,1].flatten()
    v = hsv_lower[:,:,2].flatten()
    for i in range(len(h)):
        if h[i] == -1 or s[i] == -1 or v[i] == -1:
            continue
        elif h[i] <= 180 and s[i] <= 128 and v[i] <= 128: #黑煙範圍(0,0,0)~(180,128,128) h[i] < 180 and s[i] < 255 and v[i] < 100:
            count += 1
    if count >= mask.sum() / 2:
        return True
    return "smoky" #如果不是事件，則回傳Smoky字串，目的是為了將誤判畫成白煙

# 火焰大小比對
def fireSizeFilter(ccd_uuid,size):
    if ccd_uuid == VAR.CPC or VAR.FORMOSAPLASTICS:
        if size > 1500:
            return True
    else:
        if size > 50:
            return True
    return False

# 火焰顏色比對
def compareColorFire(mask,img,model_type):
    # 顏色比對
    mask_r = np.array(np.where(mask > 0,img[:,:,0].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_g = np.array(np.where(mask > 0,img[:,:,1].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_b = np.array(np.where(mask > 0,img[:,:,2].reshape(img[:,:,2].shape[0],img[:,:,2].shape[1],1),None))
    mask_r = mask_r[mask_r != None]
    mask_g = mask_g[mask_g != None]
    mask_b = mask_b[mask_b != None]
    count = 0
    bgr_value = np.array(np.dstack((mask_b,mask_g,mask_r)),dtype='uint8')
    hsv_lower = cv2.cvtColor(bgr_value,cv2.COLOR_BGR2HSV)
    h = hsv_lower[:,:,0].flatten()
    s = hsv_lower[:,:,1].flatten()
    v = hsv_lower[:,:,2].flatten()
    if model_type == "morning":
        s_value = 34
        h_value = 156
    else:
        s_value = 0
        h_value = 171
    for i in range(len(h)):
        if h[i] == -1 or s[i] == -1 or v[i] == -1:

            continue
        elif(( h[i] <= 34  and h[i] >= 0) or (h[i] <= 180  and h[i] >= h_value)) and (s[i] <= 255 and s[i] >= s_value) and (v[i] <= 255 and v[i] >= 46):
            count += 1
        #else:
            #   print("h:",h[i]," s:",s[i]," v:",v[i])
    #print("count: ",count_test)
    #print("all: ",mask.sum())
    #print("count: ",count)
    if count >= mask.sum() / 2:
        return True
    return False

# 雙鏡頭監測項目比對
def DualCameraFilter(ccd_uuid,true_time,waitminute): 

    now_time = true_time.strftime("%Y%m%d%H%M%S")
    current_time = true_time # 現在的時間
    # check which cctv and choose json file
    if ccd_uuid == VAR.SORTH or ccd_uuid == VAR.SHANWEI:
        eventConditionJsonName = "eventCondition.json"
    elif ccd_uuid == VAR.WUFU or ccd_uuid == VAR.MOVE_ASIAPOLUMER:
        eventConditionJsonName = "eventConditionWA.json"
    elif ccd_uuid == VAR.MOVE_CPC or ccd_uuid == VAR.NORTH:
        eventConditionJsonName = "eventConditionNC.json"

    # 如果沒有json檔，則創建json檔
    if not os.path.exists(eventConditionJsonName):

        if ccd_uuid == VAR.SORTH or ccd_uuid == VAR.SHANWEI:
            eventTimeJson = {VAR.SHANWEI:[now_time,"","false"],VAR.SORTH:[now_time,"","false"]}
        elif ccd_uuid == VAR.WUFU or ccd_uuid == VAR.MOVE_ASIAPOLUMER:
            eventTimeJson = {VAR.WUFU:[now_time,"","false"],VAR.MOVE_ASIAPOLUMER:[now_time,"","false"]}
        elif ccd_uuid == VAR.MOVE_CPC or ccd_uuid == VAR.NORTH:
            eventTimeJson = {VAR.MOVE_CPC:[now_time,"","false"],VAR.NORTH:[now_time,"","false"]}

        with open(eventConditionJsonName,"w") as f:
            json.dump(eventTimeJson,f)
    
    # 打開json檔
    if ccd_uuid == VAR.SORTH or ccd_uuid == VAR.SHANWEI:
        eventConditionJson = json.load(open("eventCondition.json"))
    elif ccd_uuid == VAR.WUFU or ccd_uuid == VAR.MOVE_ASIAPOLUMER:
        eventConditionJson = json.load(open("eventConditionWA.json"))
    elif ccd_uuid == VAR.MOVE_CPC or ccd_uuid == VAR.NORTH:
        eventConditionJson = json.load(open("eventConditionNC.json"))

    # 如果是雙鏡頭，則紀錄事件當下的時間
    if ccd_uuid in VAR.isDualCamera: 
        
        #之前事件的時間
        lastEventTime = datetime.datetime.strptime(eventConditionJson[ccd_uuid][0],"%Y%m%d%H%M%S") 
        # 如果超過等待時間，則視為新事件，因此設為true
        if (current_time - lastEventTime).seconds > int(waitminute):
            eventConditionJson[ccd_uuid][2] = "true"

        #紀錄事件當下的時間
        eventConditionJson[ccd_uuid][0] = now_time
        with open(eventConditionJsonName,"w") as f:
            json.dump(eventConditionJson,f)

    flag = "nothing"

    # 取得另一個攝像頭的事件時間
    # 中日化南和汕尾
    if ccd_uuid == VAR.SORTH:
        lastEventTime = eventConditionJson[VAR.SHANWEI][0]    
    elif ccd_uuid == VAR.SHANWEI:
        lastEventTime = eventConditionJson[VAR.SORTH][0]
    # 五福和移動站塑
    elif ccd_uuid == VAR.WUFU:
        lastEventTime = eventConditionJson[VAR.MOVE_ASIAPOLUMER][0]
    elif ccd_uuid == VAR.MOVE_ASIAPOLUMER:
        lastEventTime = eventConditionJson[VAR.WUFU][0]
    # 中日化北和移動站中油
    elif ccd_uuid == VAR.MOVE_CPC:
        lastEventTime = eventConditionJson[VAR.NORTH][0]
    elif ccd_uuid == VAR.NORTH:
        lastEventTime = eventConditionJson[VAR.MOVE_CPC][0]

    last_time = datetime.datetime.strptime(lastEventTime,"%Y%m%d%H%M%S") # 另一個攝像頭的事件時間
    # 如果另一個攝像頭的事件和現在的攝像頭事件的時間在一分鐘以內，則視為同事件
    if (current_time - last_time).seconds < 60:
        flag = True
    else:
        flag = False          

    # 打開json並且將最新資料結果寫回json檔裡面
    with open(eventConditionJsonName,"w") as f:
        json.dump(eventConditionJson,f)                
    if flag == True:      
        return True
    elif flag == False:
        return False  

#### END ####

# 繪製遮罩
def DrawMask(drawed_image,mask,color):
    mask = mask*1
    mask = np.array(mask,np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    drawed_image = cv2.drawContours(drawed_image, contours, -1, color, 1)
    return drawed_image

# 檢查是否為事件
def CheckEvent(r,preprocessed_image,rects,ccd_uuid,item_name,dual_target,true_time,waitminute,model_type):
    rois = r['rois'] # 取得邊界框
    class_ids = r['class_ids'] # 取得類別
    masks = r['masks'] # 取得遮罩
    scores = r['scores'] # 取得分數
    box_in = 0
    if model_type == "morning":
        chimney = 1
        smoky = 4
        smokybk = 2
        fire = [3,5,6]
        other = 7
        windmill = 8
        tree = 9
        ironframe = 10
        light = 99
        moon = 99
        score_condition = 0.9
    elif model_type == "night":
        chimney = 1
        smoky = 2
        fire = [3,4]
        other = 99
        windmill = 99
        smokybk = 99
        tree = 99
        light = 5
        ironframe = 6
        moon = 7
        score_condition = 0.8
    for i in range(len(rois)):

        if ( class_ids[i] == smokybk or class_ids[i] in fire ) and scores[i] > score_condition:
            mask = np.where(np.array(masks[:,:,i]*1,np.uint8) != 0) # 取得有被遮罩(事件)的座標
            mask2 = masks[:,:,i] # 取得遮罩的陣列形狀
            mask2 = mask2.reshape(mask2.shape[0],mask2.shape[1],1) # 轉換為三維

            # 黑煙顏色比對條件
            if class_ids[i] == smokybk: 
                colorCompareResultSmoky = compareColor(mask2,preprocessed_image) 
                if colorCompareResultSmoky  != True:
                    return colorCompareResultSmoky 
            # 火焰大小比對的條件
            if class_ids[i] in fire:
                fireFilter = fireSizeFilter(ccd_uuid,masks[:,:,i].sum())
                if fireFilter == False:
                    return False
            # 火焰顏色比對條件
            if class_ids[i] in fire: 
                colorCompareResultFire = compareColorFire(mask2,preprocessed_image,model_type) 
                if colorCompareResultFire != True:
                    return colorCompareResultFire         
            # 旋轉鏡頭條件
            if ccd_uuid in VAR.isFixed: 
                rectFlag = "False"
                # 警示框條件
                for rect in rects: 
                    box_ins = []
                    for point_x,point_y in zip(mask[1],mask[0]): # 是否有在警示框的條件
                        box_in = cv2.pointPolygonTest(rect,(int(point_x),int(point_y)),False) # 1:在框內 0:在框邊界 -1:在框外
                        if box_in == 0:
                            box_ins.append(box_in)
                        else:
                            box_ins.append(int(math.copysign(1,box_in)))
                    
                    # 如果是火焰類別
                    if class_ids[i] in fire:
                        # 火焰條件: 有穿過警示框
                        if 1 in box_ins:
                            # 火焰條件: 有穿過警示框(同時在框內和框外)
                            if 0 in box_ins and 1 in box_ins and -1 in box_ins:
                                rectFlag = "True"
                                break
                            # 火焰條件失敗，但是火焰有在警示框內，因此判斷為False(已確定在監測項目)
                            else:
                                rectFlag = "False"
                                break
                        else:
                            # 如果火焰沒有在警示框內，則判斷為旋轉條件(發送監測項目:無)
                            rectFlag = "rotate"
                    # 如果是黑煙類別
                    else:
                        # 黑煙條件: 有經過警示框(不必穿過)
                        if 1 in box_ins:                             
                            rectFlag = "True"
                        else:
                            # 如果黑煙沒有經過警示框內，則判斷為旋轉條件(發送監測項目:無)
                            rectFlag = "rotate"
                # 雙鏡頭條件
                if ccd_uuid in VAR.isDualCamera and rectFlag != "rotate":
                    # 判斷是否有監測項目
                    if item_name in dual_target:
                        dual_flag = DualCameraFilter(ccd_uuid,true_time,waitminute)
                        if dual_flag == False:
                            return "dualEvent"
                    
                # 警示框條件沒通過但其他條件通過
                if rectFlag == "rotate": 
                    return "rectFalse"
                # 警示框條件沒通過(火焰沒穿過警示框)
                elif rectFlag == "False":
                    return False
            

            # 如果有火和煙的情況同時發生，發布時以火焰為主
            #if len(set(fire).intersection(set(class_ids))) > 0 and class_ids[i] == smokybk: 
            #    continue
            return True
# 取得指定時間的當前正確時間
def ToRealTime(time,add_time):
    diff_time = datetime.timedelta(hours=8)
    true_time = time+diff_time
    new_time = true_time + datetime.timedelta(minutes=add_time)
    return new_time
# 取得當前正確時間
def ToSpecifyTime(time):
    diff_time = datetime.timedelta(hours=8)
    true_time = datetime.datetime.now()+diff_time
    new_time = datetime.datetime.strptime(str(true_time.date())  + time,"%Y-%m-%d%H:%M")
    return new_time

# store time when event happen
def EventTime(image,true_time,ccd_uuid,waitminute,isDual): 
    
    # get event folder
    event_dir = os.path.join("history",ccd_uuid)
    today_time = time.strftime("%Y%m%d", time.localtime())
    today_event_dir = os.path.join(event_dir,today_time)
    
    # create event folder when no exist
    if not os.path.isdir(today_event_dir):
        os.makedirs(today_event_dir)
    
    json_dir = os.path.join(event_dir,today_time,"event_log.json")
    
    # create json file when no exist
    if not os.path.isfile(json_dir):

        if isDual == False:
            # 先建立事件次數json檔
            EventCount(ccd_uuid,true_time)
            # 先讀取事件次數
            event_dir = os.path.join("history",ccd_uuid)
            countJsonDir = os.path.join(event_dir,today_time,"eventResultCount.json")
            countJson = json.load(open(countJsonDir))
            count = countJson["count"]
            # 如果未連續達到兩次事件，則不進行事件紀錄
            if count != VAR.COUNT:
                # save first event image
                SaveFirstEventImage(image,ccd_uuid,true_time)
                return False          
                    
        event_json = {"time":[true_time.strftime("%Y%m%d%H%M%S")]}
        with open(json_dir,"w") as f:
            json.dump(event_json,f)
        return True
    else:
        history_json = json.load(open(json_dir))
        last_time = datetime.datetime.strptime(history_json["time"][-1],"%Y%m%d%H%M%S")

        # check event time is over 3 minutes
        if true_time > last_time and (true_time - last_time).seconds > int(waitminute) *60:
            if isDual == False:
                EventCount(ccd_uuid,true_time)

                # 先讀取事件次數
                event_dir = os.path.join("history",ccd_uuid)
                countJsonDir = os.path.join(event_dir,today_time,"eventResultCount.json")
                countJson = json.load(open(countJsonDir))
                count = countJson["count"]
                # 如果未連續達到兩次事件，則不進行事件紀錄
                if count != VAR.COUNT:
                    # save first event image
                    SaveFirstEventImage(image,ccd_uuid,true_time)
                    return False

                # 重新計算事件次數
                EventCount(ccd_uuid,true_time)
            # add event time to json file
            time_data = history_json["time"]
            time_data.append(true_time.strftime("%Y%m%d%H%M%S"))
            history_json = {"time":time_data}
            with open(json_dir,"w") as f:
                json.dump(history_json,f)
            return True

    return False
# 計算事件次數，如果連續有三個事件，則進行事件紀錄
def EventCount(ccd_uuid,true_time):
    today_time = true_time.strftime("%Y%m%d") # time for year,month,day
    event_dir = os.path.join("history",ccd_uuid)
    countJsonDir = os.path.join(event_dir,today_time,"eventResultCount.json")

    if not os.path.isfile(countJsonDir):
        countJson = {"count":1,"eventTime":true_time.strftime("%Y%m%d%H%M%S")}
        with open(countJsonDir,"w") as f:
            json.dump(countJson,f)

    else:
        history_countJson = json.load(open(countJsonDir))
        count = history_countJson["count"]
        if(true_time - datetime.datetime.strptime(history_countJson["eventTime"],"%Y%m%d%H%M%S")).seconds > 60:
            history_countJson["count"] = 1
            history_countJson["eventTime"] = true_time.strftime("%Y%m%d%H%M%S")

        elif(count != VAR.COUNT):
            history_countJson["count"] = history_countJson["count"] + 1
        else:
            history_countJson["count"] = 0
        with open(countJsonDir,"w") as f:
            json.dump(history_countJson,f)
def PredictImage(image,params_property):

    # 模型以及模型圖和模型session
    global model,sess,graph

    # 紀錄事件的時間以及資料
    def SaveEventResult():
        event_dir = os.path.join("history",ccd_uuid)
        resultJsonDir = os.path.join(event_dir,today_time,"eventResult.json")
        resultJson = {true_time.strftime("%Y%m%d%H%M%S"):{"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)}}
        if not os.path.isfile(resultJsonDir):
            with open(resultJsonDir,"w") as f:
                json.dump(resultJson,f)

        else:
            history_json = json.load(open(resultJsonDir))
            history_json[true_time.strftime("%Y%m%d%H%M%S")] = {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)}
            with open(resultJsonDir,"w") as f:
                json.dump(history_json,f)

    # 紀錄事件的時間以及資料到logs資料夾
    def SaveEventLog():
        dayLog_dir = os.path.join("logs",ccd_uuid,today_time,"jsons")
        if "fire" in result_data["label"]:
            log_dir = os.path.join(dayLog_dir,"fire_log.json")
        else:
            log_dir = os.path.join(dayLog_dir,"smoky_log.json")
        
        resultJson = {true_time.strftime("%Y%m%d%H%M%S"):{"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)}}
        if not os.path.isfile(log_dir):
            with open(log_dir,"w") as f:
                json.dump(resultJson,f)
        else:
            history_json = json.load(open(log_dir))
            history_json[true_time.strftime("%Y%m%d%H%M%S")] = {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)}
            with open(log_dir,"w") as f:
                json.dump(history_json,f)

    def PredictingImage(image): # model predicting
        
        with graph.as_default(): # model detect in same graph when sending image
            tf.compat.v1.keras.backend.set_session(sess)
            detected_results = model.detect([image],verbose=0)[0] # Run detection

        return detected_results 

    def ImagePostprocessing(detect_result,image,origin_image,ccd_tags,dual_target,ccd_uuid,true_time): # image postprocessing
        # 變數設定
        drawed_image = cv2.cvtColor(origin_image.copy(),cv2.COLOR_BGR2RGB) # 用來畫圖的影像
        preprocessed_image = origin_image #cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB) # 用來做後處理判斷的影像
        rois = detect_result["rois"] # 取得邊界框
        class_ids = detect_result["class_ids"] # 取得類別
        scores = detect_result["scores"] # 取得分數
        masks = detect_result["masks"] # 取得遮罩
        event_temp = [] # 事件暫存
        result_temp = [] # 結果暫存
        dual_temp = [] # 雙鏡頭暫存
        db_label = [] # 事件名稱暫存
        db_notify_info = [] # 事件資訊暫存
        isDual = False # 是否有雙鏡頭事件
        isDualTarget = False # 是雙鏡頭監測項目是否有監測到事件
        # 夜晚以及白天的模型設定
        if model_type == "morning":
            text_color = (0,0,0) # 文字顏色
            normal_labels = VAR.NORMAL_MORNING # 早晨模型的類別
            class_name_id = VAR.class_names_morning # 早晨模型的類別名稱
            label_dict = VAR.EVENT_MORNING # 事件字典
            label_dict_regular = VAR.EVENT_MORNING_REGULAR # 事件字典(火焰事件統一名稱)
            score_condition = 0.9 # 分數條件
            ironframe = 99
            ironframerectangle = 99
        else:
            text_color = (255,255,255)
            class_name_id = VAR.class_names_night
            normal_labels = VAR.NORMAL_NIGHT # 夜晚模型的類別
            label_dict = VAR.EVENT_NIGHT
            label_dict_regular = VAR.EVENT_NIGHT_REGULAR # 事件字典(火焰事件統一名稱)
            ironframe = 10
            ironframerectangle = 11
            score_condition = 0.8
        hasEvent = False # 是否有事件
        eventKind = "" # 事件種類

        # 繪製邊界框、文字、儲存結果
        for i in range(len(ccd_tags)):
            
            # 如果有監測項目並且有監測框
            if "START" in ccd_tags[i]:
                # 繪製監測項目邊界框
                x1,y1,x2,y2,drawed_image = DrawBoundingBox(ccd_tags[i],origin_shape,drawed_image)
                rect = [np.array([np.array([x1,y1]),np.array([x2,y1]),np.array([x2,y2]),np.array([x1,y2])])]              
                drawed_image = cv2ImgAddText(drawed_image,ccd_tags[i]["NAME"],(x1+(x2 - x1)/2,y2+10),text_color)
                rects.append(np.array([np.array([x1,y1]),np.array([x2,y1]),np.array([x2,y2]),np.array([x1,y2])]))
                # 繪製以及判斷事件
                for j in range(len(rois)):
                    mask = masks[:,:,j] # 取得遮罩的陣列形狀
                    draw_flag = False # 繪製的flag

                    # 繪製煙囪
                    if class_ids[j] == normal_labels["chimney"]:
                        draw_flag = True
                        color = (19,69,139) # brown bgr(opencv)

                    # 模型判斷為事件並且分數大於設定的條件
                    if (class_ids[j] in label_dict.keys()) and scores[j] > score_condition:

                        # 判斷是否為事件
                        isEvent = CheckEvent({"rois":[rois[j]],"class_ids":[class_ids[j]],"scores":[scores[j]],"masks":masks[:,:,j].reshape(576,1024,1)},preprocessed_image,rect,ccd_uuid,ccd_tags[i]["NAME"],dual_target,true_time,waitminute,model_type)
                        event_temp.append(str(isEvent)) # 事件暫存
                        result_temp.append(str(class_ids[j])) # 結果暫存
                        # 如果是事件
                        if isEvent == True:
                            # 如果攝影機是雙鏡頭並且有事件，但是監測項目監測框無事件但其他監測框有事件，則進行事件紀錄
                            isDualTarget = False

                            # 事件list裡面還沒有這個監測項目的NOTE
                            if not ccd_tags[i]["NOTE"] in db_notify_info:
                                # 紀錄事件
                                db_notify_info.append(ccd_tags[i]["NOTE"])
                                db_label.append(label_dict_regular[class_ids[j]])

                                # 如果是雙鏡頭並且有雙鏡頭的監測項目
                                if ccd_uuid in VAR.isDualCamera and ccd_tags[i]["NAME"] in dual_target:
                                    dual_temp.append(True)
                                else:
                                    dual_temp.append(False)

                        elif isEvent == "rectFalse":
                            # 是否有事件，就算只有一個事件也算
                            hasEvent = True 
                            eventKind = label_dict_regular[class_ids[j]]
                        elif isEvent == "dualEvent":
                            # 如果攝影機是雙鏡頭，則當監測項目監測框有事件，則不進行「無」的事件紀錄
                            isDualTarget = True

                        # 計算影像比例
                        x_ratio = rois[j][3] - rois[j][1] #int(1200*(rois[i][3] - rois[i][1])/image.shape[0])
                        y_ratio = rois[j][2] - rois[j][0] #int(1000*(rois[i][2] - rois[i][0])/image.shape[1])

                        # 繪製白煙以及事件
                        if isEvent == "smoky" or isEvent == False:
                            draw_flag = True
                            color = (255,0,0)
                        elif isEvent == True or isEvent == "rectFalse":
                            draw_flag = True
                            color = (0,0,255)
                            # 繪製事件名稱
                            cv2.putText(drawed_image,class_name_id[class_ids[j]],(int(rois[j][1] + x_ratio/3),int(rois[j][2] - y_ratio/20)),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1) 
                        
                        # 繪製遮罩
                        if draw_flag != False:
                            drawed_image = DrawMask(drawed_image,mask,color)


            elif ccd_uuid in VAR.isRotation:
                for j in range(len(rois)):

                    mask = masks[:,:,j] # 取得遮罩的陣列形狀
                    draw_flag = False # 繪製的flag
                    # 繪製煙囪
                    if class_ids[j] == normal_labels["chimney"]:
                        draw_flag = True
                        color = (19,69,139) # brown bgr(opencv)

                    #如果攝像頭是台塑，然後有偵測到ironframe，就跳過
                    if ccd_uuid == VAR.FORMOSAPLASTICS:
                        if(class_ids[i] == ironframe or ironframe in class_ids):
                            break     
                        #20240914 louis 修正；如果攝像頭是台塑，然後有偵測到ironframerectangle，就跳過
                        if(class_ids[i] == ironframerectangle or ironframerectangle in class_ids):
                            break 
                    # 模型判斷為事件並且分數大於設定的條件
                    if (class_ids[j] in label_dict.keys()) and scores[j] > score_condition:
                        rect = [np.array([0,0,0,0])]
                        isEvent = CheckEvent({"rois":[rois[j]],"class_ids":[class_ids[j]],"scores":[scores[j]],"masks":masks[:,:,j].reshape(576,1024,1)},preprocessed_image,rect,ccd_uuid,ccd_tags[i]["NAME"],dual_target,true_time,waitminute,model_type)

                        # 如果是事件並且事件list裡面還沒有這個監測項目的NOTE
                        if not ccd_tags[i]["NOTE"] in db_notify_info and isEvent == True:
                            db_notify_info.append(ccd_tags[i]["NOTE"])
                            db_label.append(label_dict_regular[class_ids[j]])

                        # 計算影像比例
                        x_ratio = rois[j][3] - rois[j][1] #int(1200*(rois[i][3] - rois[i][1])/image.shape[0])
                        y_ratio = rois[j][2] - rois[j][0] #int(1000*(rois[i][2] - rois[i][0])/image.shape[1])

                        # 繪製白煙以及事件
                        if isEvent == "smoky" or isEvent == False:
                            draw_flag = True
                            color = (255,0,0)
                        elif isEvent == True or isEvent == "rectFalse":
                            draw_flag = True
                            color = (0,0,255)
                            # 繪製事件名稱
                            cv2.putText(drawed_image,class_name_id[class_ids[j]],(int(rois[j][1] + x_ratio/3),int(rois[j][2] - y_ratio/20)),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1) 
                        
                    # 繪製遮罩
                    if draw_flag != False:
                        drawed_image = DrawMask(drawed_image,mask,color)
        # 如果有事件並且事件list是空的和是固定鏡頭
        if hasEvent == True and len(db_notify_info) == 0 and ccd_uuid in VAR.isFixed and isDualTarget == False:
            rule_SN = ccd_tags[0]["NOTE"].split(",")[0] # 取得規則SN
            ccd_SN = ccd_tags[0]["NOTE"].split(",")[1] # 取得CCD SN
            db_notify_info.append(rule_SN + "," + ccd_SN + ",無,31")
            db_label.append(eventKind)

        # 如果有事件並且事件list不是空的和是雙鏡頭，雙鏡頭暫存裡面沒有非雙鏡頭監測項目
        if ccd_uuid in VAR.isDualCamera and len(db_notify_info) > 0 and db_notify_info[0].split(",")[2] != "無" and False not in dual_temp:

            # 紀錄雙鏡頭事件
            if (ccd_uuid == VAR.SORTH or ccd_uuid == VAR.SHANWEI): 
                eventConditionJsonName = "eventCondition.json"
            elif (ccd_uuid == VAR.NORTH or ccd_uuid == VAR.MOVE_CPC):
                eventConditionJsonName = "eventConditionNC.json"
            elif (ccd_uuid == VAR.WUFU or ccd_uuid == VAR.ASIAPOLUMER):
                eventConditionJsonName = "eventConditionWA.json"
            
            # 找到是屬於雙鏡頭監測項目的事件
            dual_db_notify_info = [db_notify_info[i] for i, x in enumerate(dual_temp) if x == True]

            # 打開json並且將最新資料結果寫回json檔裡面
            eventConditionJson = json.load(open(eventConditionJsonName))
            eventConditionJson[ccd_uuid][0] = now_time
            eventConditionJson[ccd_uuid][1] = dual_db_notify_info
            with open(eventConditionJsonName,"w") as f:
                json.dump(eventConditionJson,f)
            isDual = True
        return drawed_image,db_notify_info,db_label,isDual

    ####################### time data #######################
    diff_time =datetime.timedelta(hours=8) # gis docker time need add 8 hours
    true_time = datetime.datetime.now() + diff_time # real time
    now_time = true_time.strftime("%Y%m%d%H%M%S") # time for year,month,day,hour,minute,second
    today_time = true_time.strftime("%Y%m%d") # time for year,month,day
    now_time_datetimeformat = datetime.datetime.strptime(now_time,"%Y%m%d%H%M%S")
    sunrise_time,sunset_time = GetSundata(true_time) # get sunrise and sunset time
    #sunrise_time,sunset_time = "06:40","17:30"
    sunset_time = (datetime.datetime.strptime(sunset_time.replace(":",""),"%H%M") + datetime.timedelta(minutes=30)).strftime("%H:%M") #- datetime.timedelta(minutes=30)).strftime("%H:%M")  sunset time - 30 minutes
    sunrise_time = (datetime.datetime.strptime(sunrise_time.replace(":",""),"%H%M") - datetime.timedelta(minutes=30)).strftime("%H:%M") # sunrise time - 30 minutes
    night_time = datetime.datetime(true_time.year,true_time.month,true_time.day,int(sunset_time.split(":")[0]),int(sunset_time.split(":")[1]))  # 夜晚時間(完整時間格式)
    morning_time = datetime.datetime(true_time.year,true_time.month,true_time.day,int(sunrise_time.split(":")[0]),int(sunrise_time.split(":")[1]))  # 白天時間(完整時間格式)
    ####################### end #######################

    ####################### paremeters #######################

    # request param
    text = ""
    en_keepimg = False
    event_alert = "False" # 是否有事件，根據此參數決定是否發送警告
    ccd_uuid = "xxuuidxx"
    ccd_tags = None
    waitminute = "10"
    db_label = []
    db_notify_info = []
    #20241104新增:雙鏡頭標的
    dual_target = ""

    # other param
    rects = list([np.array([np.array([0,0]),np.array([0,0]),np.array([0,0])])])
    
    # morning model or night model
    with open("model_type.txt","r") as f: 
        model_type = f.read()

    # check parameter exist ,if not, set value to blank
    en_keepimg = CheckParamExist("EN_KEEPIMG", params_property)
    ccd_uuid = CheckParamExist("UUID", params_property)
    ccd_tags = CheckParamExist("TAGS", params_property)
    waitminute = CheckParamExist("WAIT_MINUTE", params_property)
    #20241104新增:雙鏡頭標的
    dual_target = CheckParamExist("DUAL", params_property)
    if dual_target == None:
        dual_target = []
    ccd_uuid_name = ccd_uuid[:8]
    
    text += "CCD: " + ccd_uuid + " "

    ####################### end #######################

    ####################### model loading #######################

    if true_time > morning_time and true_time < night_time and model_type == "night": #Change model base on time
        # change model type to morning
        model_type = "morning"
        WriteModelType(model_type)
        model = ChangeModel(VAR.MODEL_DIR_MORNING,model_type)        
    elif (true_time < morning_time or true_time > night_time) and model_type == "morning":
        # change model type to night
        model_type = "night"
        WriteModelType(model_type)
        model = ChangeModel(VAR.MODEL_DIR_NIGHT,model_type)

    # when changing model, return origin image,avoid error
    if (true_time - morning_time).seconds < 60 or (true_time - night_time).seconds < 60:
        imagebase64 = Image2Base64(image)

        return {"imageBase64":imagebase64.decode("utf-8"), "result": "", "alert": "False"},ccd_uuid_name+now_time+".jpg",now_time_datetimeformat
    
    ####################### end #######################

    ####################### predict image #######################
    origin_shape = [image.shape[0],image.shape[1]] # origin image size   
    resized_image = cv2.resize(image,VAR.IMAGE_RESIZE, interpolation=cv2.INTER_AREA) # resize image

    # turn RGB to BGR
    #resized_image = cv2.cvtColor(resized_image,cv2.COLOR_RGB2BGR) 
    # do image preprocess   
    preprocessed_image = ImagePreprocessing(resized_image) 
    # detecting image by model
    detect_result = PredictingImage(preprocessed_image) 

    # turn RGB to BGR
    preprocessed_image = cv2.cvtColor(preprocessed_image,cv2.COLOR_RGB2BGR)
    # 影像後處理，包含繪製邊界框、文字、儲存結果
    postprocessed_image,db_notify_info,db_label,isDual = ImagePostprocessing(detect_result,preprocessed_image,resized_image,ccd_tags,dual_target,ccd_uuid,true_time)
    # 繪製模型時間
    model_text = "AM" if model_type == "morning" else "PM"
    # draw model type in right corner
    postprocessed_image = cv2.rectangle(postprocessed_image,(int(postprocessed_image.shape[1] * 0.95),int(postprocessed_image.shape[0] * 0.95)),(int(postprocessed_image.shape[1] * 0.988),int(postprocessed_image.shape[0] * 0.99)),(0,0,0),-1)
    postprocessed_image = cv2ImgAddText(postprocessed_image,model_text,(postprocessed_image.shape[1] * 0.955,postprocessed_image.shape[0] * 0.94),(255,255,255))
    
    # convert image to base64
    postprocessed_image = cv2.cvtColor(postprocessed_image,cv2.COLOR_BGR2RGB) # convert BGR to RGB
    imagebase64 = Image2Base64(postprocessed_image)

    # save all_log.json to database when 23:58~23:59
    if ToSpecifyTime("23:59") > ToRealTime(datetime.datetime.now(),0) and ToRealTime(datetime.datetime.now(),0) > ToSpecifyTime("23:58"):
        # get json directory
        jsonDir = os.path.join("logs",ccd_uuid,now_time_datetimeformat.strftime("%Y%m%d"),"jsons")
        jsonDir_All = os.path.join(jsonDir,"all_log.json")
        jsonInfo_All = json.load(open(os.path.join(jsonDir_All)))

        count_all = len(jsonInfo_All) # get count of detection
        result_data = { "classes": str(detect_result['class_ids']), "scores": str(detect_result['scores']), "notify": db_notify_info, "label":db_label, "waitminute":str(waitminute),"LINE_NOTIFY_TIME": true_time.strftime("%Y-%m-%d %H:%M:%S"),"count":count_all,"flag":"true"}    
        return {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": "True"},ccd_uuid_name+now_time+".jpg",true_time
    else:
        result_data = { "classes": str(detect_result['class_ids']), "scores": str(detect_result['scores']), "notify": db_notify_info, "label":db_label, "waitminute":str(waitminute),"LINE_NOTIFY_TIME": true_time.strftime("%Y-%m-%d %H:%M:%S")}
    
    # 測試的時候顯示圖片
    #plt.show()
    #plt.imshow(postprocessed_image)

    # 如果有事件
    if len(db_notify_info) > 0:
        
        postprocessed_image = cv2.cvtColor(postprocessed_image,cv2.COLOR_RGB2BGR) # convert RGB to BGR
        event_alert = EventTime(postprocessed_image,true_time,ccd_uuid,waitminute,isDual)
        text += "event_alert_time_stop: " + str(event_alert) + " " # record text

        # 儲存暫存影像
        savePath = os.path.join("temp")
        imageSaveName = os.path.join(savePath,ccd_uuid + ".jpg")
        os.makedirs(savePath,exist_ok=True)
        cv2.imwrite(imageSaveName,postprocessed_image)         
        # 如果事件發送條件通過
        if event_alert == True:
            store_eventImg_path = os.path.join("event_images")
            for store_destination in ["origin","output"]:
                output_tempfullpath = os.path.join(store_eventImg_path,store_destination,ccd_uuid_name+now_time+".jpg")
                if store_destination == "output":
                    stored_image = postprocessed_image.copy() # 儲存事件影像
                else:
                    stored_image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # 儲存原始影像
                cv2.imwrite(output_tempfullpath,stored_image)

                # 儲存事件影像在logs資料夾
                if "fire" in result_data["label"]:
                    image_dir = os.path.join("logs",ccd_uuid,today_time,"event","fire",now_time+".jpg")
                else:
                    image_dir = os.path.join("logs",ccd_uuid,today_time,"event","smoky",now_time+".jpg")
                cv2.imwrite(image_dir,stored_image)          

            
            # 20240723 louis 更新
            if isDual == True:
                if ccd_uuid == VAR.SORTH or ccd_uuid == VAR.SHANWEI:
                    eventConditionJson = json.load(open("eventCondition.json"))
                    eventConditionJson_name = "eventCondition.json"
                elif ccd_uuid == VAR.WUFU or ccd_uuid == VAR.MOVE_ASIAPOLUMER:
                    eventConditionJson = json.load(open("eventConditionWA.json"))
                    eventConditionJson_name = "eventConditionWA.json"
                elif ccd_uuid == VAR.MOVE_CPC or ccd_uuid == VAR.NORTH:
                    eventConditionJson = json.load(open("eventConditionNC.json"))
                    eventConditionJson_name = "eventConditionNC.json"
                # 攝影機是雙鏡頭條件時
                if ccd_uuid in VAR.isDualCamera:
                    
                    # 中日化南和汕尾
                    if ccd_uuid == VAR.SORTH:
                        another_ccd_uuid = VAR.SHANWEI
                    elif ccd_uuid == VAR.SHANWEI:
                        another_ccd_uuid = VAR.SORTH
                    # 五福和移動站塑
                    elif ccd_uuid == VAR.WUFU:
                        another_ccd_uuid = VAR.MOVE_ASIAPOLUMER
                    elif ccd_uuid == VAR.MOVE_ASIAPOLUMER:
                        another_ccd_uuid = VAR.WUFU
                    # 中日化北和移動站中油
                    elif ccd_uuid == VAR.MOVE_CPC:
                        another_ccd_uuid = VAR.NORTH
                    elif ccd_uuid == VAR.NORTH:
                        another_ccd_uuid = VAR.MOVE_CPC     

                    # 開啟json並且將狀態設為true(等待發送)
                    eventConditionJson[ccd_uuid][2] = "true"
                    with open(eventConditionJson_name,"w") as f:
                        json.dump(eventConditionJson,f)           

                    # 另一個攝像頭的事件
                    lastEventTime = eventConditionJson[another_ccd_uuid][0]
                    lastEvent = eventConditionJson[another_ccd_uuid][1]
                    lastSend = eventConditionJson[another_ccd_uuid][2]

                    # 取得前一次事件的時間
                    event_dir = os.path.join("history",another_ccd_uuid)
                    today_time = time.strftime("%Y%m%d", time.localtime())
                    today_event_dir = os.path.join(event_dir,today_time)
                    json_dir = os.path.join(event_dir,today_time,"event_log.json")
                    history_json = json.load(open(json_dir))
                    last_time = datetime.datetime.strptime(history_json["time"][-1],"%Y%m%d%H%M%S")
                
                    # 取得另一個攝像頭的temp的影像
                    output_tempfullpath2 = os.path.join("temp",another_ccd_uuid+".jpg")

                    #這次與前一次事件的時間比對(另一個攝像頭的事件)
                    if (true_time - last_time).seconds > int(waitminute) *60 and lastSend == "true":
                        eventConditionJson[ccd_uuid][2] = "false"
                        eventConditionJson[another_ccd_uuid][2] = "false"
                        Line_Notify(output_tempfullpath,true_time, ccd_uuid,db_notify_info)
                        Line_Notify(output_tempfullpath2,datetime.datetime.strptime(lastEventTime,"%Y%m%d%H%M%S"), another_ccd_uuid,lastEvent)
                        with open(eventConditionJson_name,"w") as f:
                            json.dump(eventConditionJson,f)
                    # 若事件沒有監測項目，則採用單鏡頭規則發送
                    else:
                        Line_Notify(output_tempfullpath,true_time, ccd_uuid,db_notify_info)
            else:
                Line_Notify(output_tempfullpath,true_time, ccd_uuid,db_notify_info)
            SaveEventResult()
            SaveEventLog()
            return {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)},ccd_uuid_name+now_time+".jpg",true_time
        else:
            return {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": "False"},ccd_uuid_name+now_time+".jpg",true_time
    else:

        return {"imageBase64": imagebase64.decode("utf-8"), "result":str(result_data), "alert": str(event_alert)},ccd_uuid_name+now_time+".jpg",true_time 
    ####################### end #######################


sess = tf.Session()
graph = tf.get_default_graph()
keras.backend.set_session(sess)

with open("model_type.txt","r") as f:
    model_type = f.read()
    
if model_type == "morning":
    model_dir = VAR.MODEL_DIR_MORNING
    num_classes = len(VAR.class_names_morning)
elif model_type == "night":
    model_dir = VAR.MODEL_DIR_NIGHT
    num_classes = len(VAR.class_names_night)    
    
class InferenceConfig(training_multi.SampleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES =  num_classes
    BACKBONE = "resnet101"
    DETECTION_MIN_CONFIDENCE = 0.7
        #MINI_MASK_SHAPE = (56, 56)
config = InferenceConfig()
config.display()
     
model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config) # Create model object in inference mode.
#model.keras_model.load_weights(model_dir, by_name=True) # Load weights
model.load_weights(model_dir, by_name=True)
model.keras_model._make_predict_function()
    
