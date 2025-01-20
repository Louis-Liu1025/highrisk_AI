from flask import Flask, request, jsonify, abort
from predict import PredictImage
import os
import sys
import random
#from io import ByteIO
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2
import random as rd
import argparse
from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
import variable as VAR
import json
import time
import datetime
# Root directory of the project


def MakeFolder(ccd_uuid):
    
    # param
    diff_time = datetime.timedelta(hours=8)
    true_time = datetime.datetime.now()+diff_time
    history_dir = "history"
    history_uuid_dir = os.path.join(history_dir,ccd_uuid)
    history_day_uuid_dir = os.path.join(history_uuid_dir,true_time.strftime("%Y%m%d"))
    
    detected_image_dir = "detected_images"
    detected_uuid_dir = os.path.join(detected_image_dir,ccd_uuid)
    detected_morning_origin_uuid_dir = os.path.join(detected_uuid_dir,true_time.strftime("%Y%m%d"),"morning","origin")
    detected_morning_output_uuid_dir = os.path.join(detected_uuid_dir,true_time.strftime("%Y%m%d"),"morning","output")
    detected_night_origin_uuid_dir = os.path.join(detected_uuid_dir,true_time.strftime("%Y%m%d"),"night","origin")
    detected_night_output_uuid_dir = os.path.join(detected_uuid_dir,true_time.strftime("%Y%m%d"),"night","output")
    
    event_image_dir = "event_images"
    event_origin_dir = os.path.join(event_image_dir,"origin")
    event_output_dir = os.path.join(event_image_dir,"output")

    log_dir = "logs"
    perImageInfo_dir = os.path.join(log_dir,ccd_uuid,true_time.strftime("%Y%m%d"),"jsons")
    perImage_fire_dir = os.path.join(log_dir,ccd_uuid,true_time.strftime("%Y%m%d"),"event","fire")
    perImage_smoky_dir = os.path.join(log_dir,ccd_uuid,true_time.strftime("%Y%m%d"),"event","smoky")
    
    # detected result data
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)      
    
    if not os.path.exists(history_day_uuid_dir):
        os.makedirs(history_day_uuid_dir)  
    
    # detected result image       
    if not os.path.exists(detected_morning_origin_uuid_dir):
        os.makedirs(detected_morning_origin_uuid_dir)
        
    if not os.path.exists(detected_morning_output_uuid_dir):
        os.makedirs(detected_morning_output_uuid_dir)

    if not os.path.exists(detected_night_origin_uuid_dir):
        os.makedirs(detected_night_origin_uuid_dir)
        
    if not os.path.exists(detected_night_output_uuid_dir):
        os.makedirs(detected_night_output_uuid_dir)  
        
    # event result image
    if not os.path.exists(event_origin_dir):
        os.makedirs(event_origin_dir)
        
    if not os.path.exists(event_output_dir):
        os.makedirs(event_output_dir)
    
    # count images
    if not os.path.exists(perImageInfo_dir):
        os.makedirs(perImageInfo_dir)        
    if not os.path.exists(perImage_fire_dir):
        os.makedirs(perImage_fire_dir)
    if not os.path.exists(perImage_smoky_dir):
        os.makedirs(perImage_smoky_dir)
def GetAllCount(ccd_uuid,now_time_datetimeformat,result_json):
    new_result_json = {"imageBase64":result_json["imageBase64"],"result":{"LINE_NOTIFY_TIME":ToRealTime(datetime.datetime.now(),0).strftime("%Y-%m-%d %H:%M:%S"),"count":"","flag":"","classes":"","notify":[],"scores":"","label":[],"waitminute":""},"alert": "True"}
    jsonDir = os.path.join("logs",ccd_uuid,now_time_datetimeformat.strftime("%Y%m%d"),"jsons")
    jsonDir_All = os.path.join(jsonDir,"all_log.json")
    #jsonDir_fire = os.path.join(jsonDir,"fire_log.json")
    #jsonDir_smoky = os.path.join(jsonDir,"smoky_log.json")

    jsonInfo_All = json.load(open(os.path.join(jsonDir_All)))
    #jsonInfo_fire = json.load(open(os.path.join(jsonDir_fire)))
    #jsonInfo_smoky = json.load(open(os.path.join(jsonDir_smoky)))
    count_all = len(jsonInfo_All)
    #count_fire = len(jsonInfo_fire) if os.path.exists(jsonInfo_fire) else 0
    #count_smoky = len(jsonInfo_smoky) if os.path.exists(jsonInfo_smoky) else 0

    new_result_json["result"]["count"] = count_all
    new_result_json["result"]["flag"] = "true"
    return new_result_json
def ToRealTime(time,add_time):
    diff_time = datetime.timedelta(hours=8)
    true_time = time+diff_time
    new_time = true_time + datetime.timedelta(minutes=add_time)
    return new_time
def ToSpecifyTime(time):
    diff_time = datetime.timedelta(hours=8)
    true_time = datetime.datetime.now()+diff_time
    new_time = datetime.datetime.strptime(str(true_time.date())  + time,"%Y-%m-%d%H:%M")
    return new_time
def SaveResultLog(result_json,ccd_uuid,now_time_datetimeformat,image_name,spended_time,end_time):
    
    '''
    history_day_uuid_dir = os.path.join("history",ccd_uuid,now_time_datetimeformat.strftime("%Y%m%d"))
    json_dir = os.path.join(history_day_uuid_dir,"result_log.json")
    
    if not os.path.isfile(json_dir):
        result_log_json = {image_name:result_json}
        result_log_json[image_name]["store_time"] = now_time_datetimeformat.strftime("%Y%m%d%H%M%S")
        result_log_json[image_name]["spended_time"] = spended_time
        with open(json_dir,"w") as f:
            json.dump(result_log_json,f)
    else:
        history_json = json.load(open(json_dir))
        last_time = datetime.datetime.strptime(history_json[list(history_json.keys())[-1]]["store_time"],"%Y%m%d%H%M%S")
        if now_time_datetimeformat > last_time and (now_time_datetimeformat - last_time).seconds > VAR.KEEP_IMAGE_TIME: #int(waitminute)*6:
            #event_json = {"time":time.strftime("%Y%m%d%H%M%S", time.localtime())}
            history_json[image_name] = result_json
            history_json[image_name]["store_time"] = now_time_datetimeformat.strftime("%Y%m%d%H%M%S")
            history_json[image_name]["spended_time"] = spended_time
            with open(json_dir,"w") as f:
                json.dump(history_json,f)
    '''
    # 紀錄全部影像數量
    try:
        dayLog_dir = os.path.join("logs",ccd_uuid,now_time_datetimeformat.strftime("%Y%m%d"),"jsons")
        json_dir = os.path.join(dayLog_dir,"all_log.json")
        if not os.path.exists(json_dir):
            log = {image_name:{"DATA_TIME":now_time_datetimeformat.strftime("%Y%m%d%H%M%S")}}
            with open(json_dir,"w") as f:
                json.dump(log,f)
        else:
            log = json.load(open(json_dir))
            log[image_name] = {"DATA_TIME":now_time_datetimeformat.strftime("%Y%m%d%H%M%S")}
            with open(json_dir,"w") as f:
                json.dump(log,f)       
    except Exception as e:
        json_dir = os.path.join("logs", ccd_uuid, now_time_datetimeformat.strftime("%Y%m%d"), "jsons", "all_log.json")
        if os.path.exists(json_dir):
            os.remove(json_dir)
        print("UUID:",ccd_uuid,"error:",e,"，已刪除json")

def CheckParamExist(tag,params):
    value = None
    if tag.upper() in params:
        value = params[tag.upper()]
    return value

def Action_Service(token=None):
            
    params = None
    params_property = None
    image_file = None
    try:
        params = json.loads(request.form["params"])
        
    except:
        response_message = "Params Requst Error."
        abort(404,description={"Params Requst Error."})
        
    try:
        image_file  = cv2.cvtColor(cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    except:
        response_message = "Image Request Error."
        abort(404,description={"Image Request Error."})
    
    if params is None or image_file is None:
        abort(404,description={"imageBase64": "", "result": "property format incorrect(IMAGE_SOURCE)", "alert": "False"})
    else:
        
        image_source = CheckParamExist("IMAGE_SOURCE",params)
        if image_source == None:
            abort(404, description={"imageBase64": "", "result": "property format incorrect(IMAGE_SOURCE)", "alert": "False"})
        

        params_property = params["PROPERTY"][0]
        
        ccd_uuid = CheckParamExist("UUID", params_property)
        param_name = CheckParamExist("NAME",params_property)

        if param_name == None:
            abort(404, description={"imageBase64": "", "result": "property format incorrect(NAME)", "alert": "False"})
        if ccd_uuid == None:
            abort(404, description={"imageBase64": "", "result": "property format incorrect(UUID)", "alert": "False"})
        MakeFolder(ccd_uuid)
        try:
            start_time = time.time() # time count start
            result_json,image_name,now_time_datetimeformat = PredictImage(image_file,params_property)
            end_time = time.time() # time count stop
            spended_time = format(end_time - start_time,".2f")
            
            SaveResultLog(result_json,ccd_uuid,now_time_datetimeformat,image_name,spended_time,end_time)
            
            print("UUID:",ccd_uuid,"prediction spend time: ",spended_time," seconds","\n")
        except Exception as e:
            print("UUID:",ccd_uuid,"error:",e)
            abort(404, description="something got wrong")
    #FIT.release_log(params_name, "", "", True, first_time, True, end_time, True)
    '''
    if ToSpecifyTime("17:59") > ToRealTime(datetime.datetime.now(),0) and ToRealTime(datetime.datetime.now(),0) > ToSpecifyTime("14:58"):
        new_json = result_json
        new_json["result"] = new_json["result"].replace("}","")
        new_json["result"] += ",'flag':'true'}"
        new_json["alert"] = "True"
        #new_json = GetAllCount(ccd_uuid,now_time_datetimeformat,result_json)
        print(new_json)
        return jsonify(new_json)
    else:
    '''
    return jsonify(result_json)

def Create_Service():
    diff_time =datetime.timedelta(hours=8) # gis docker time need add 8 hours
    true_time = datetime.datetime.now() + diff_time # real time
    now_time = true_time.strftime("%Y%m%d%H%M%S") # time for year,month,day,hour,minute,second
    eventConditionJsonName = "eventCondition.json"
    if not os.path.exists(eventConditionJsonName):
        eventTimeJson = {VAR.SHANWEI:[now_time,"","false"],VAR.SORTH:[now_time,"","false"]}
        with open(eventConditionJsonName,"w") as f:
            json.dump(eventTimeJson,f)
            
    app = Flask(__name__)
    port = VAR.SERVICE_PORT
    
    if VAR.SERVICE_TOKEN_MODE == False:
        @app.route('/', methods=['POST'])
        def no_token():
            return Action_Service()
        app.run(host='0.0.0.0', port=port, threaded=True)
    else:
        @app.route('/<string:token>', methods=['POST'])
        def token_mode(token):
            if token != VAR.SERVICE_TOKEN:
                abort(404,description = 'Token Not Correct')
            else:
                return Action_Service(token)
        app.run(host='0.0.0.0', port=port, threaded=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='flask test.')
    parser.add_argument("command",metavar="<command>",help="create")
    args = parser.parse_args()
    if args.command == "create":
        Create_Service()