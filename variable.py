'''
project variable
'''
import os 

############### Setting model ###############
MODEL_DIR_MORNING = os.path.join("model","maskrcnn_model_morning_20240823.h5")
MODEL_DIR_NIGHT = os.path.join("model","model20241119_night.h5")# predict model directory
class_names_morning = ['BG', "chimney","smokybk","fire","smoky","fireboom","firesmoky","other","windmill","tree","ironframe"]
class_names_night = ['BG', "chimneydk","smokydk","fire","fireshine","light","ironframe","moon"]
isRotation = ["e5cd52d4-6364-4df7-9c6f-6b66a9d66bbf","a8ffb0bd-d0cd-4783-9d1e-785e41675829","ff27c511-0974-4502-ba07-448f0f9679c7","fb5df986-fc0c-4d10-b4a4-b8af0c2f9d83"]
isFixed = ["4ab74857-9bed-4f65-afee-0a227e5b7599","824d5d8c-6319-4f91-a383-8f9c79a89ebc","6eb1f359-c37b-4f0c-a831-51420a0ca19a","d7ba310f-7158-49ce-a427-133228651b2a","65033b06-d667-45a4-a490-d295cbd14238","ebabb307-04b1-4cf2-aa61-a609d8978b0f","dbe8a60b-98af-444e-8af7-4ea7245d8c0b","e30dbdb2-b392-4fb6-9a2f-06eaf10929ff"]
isDualCamera = ["dbe8a60b-98af-444e-8af7-4ea7245d8c0b","824d5d8c-6319-4f91-a383-8f9c79a89ebc","4ab74857-9bed-4f65-afee-0a227e5b7599","65033b06-d667-45a4-a490-d295cbd14238","d7ba310f-7158-49ce-a427-133228651b2a","e30dbdb2-b392-4fb6-9a2f-06eaf10929ff"]

############### Setting parameter ###############
COUNT = 2 # 如果重複一次
EVENT_MORNING = {class_names_morning.index("smokybk"):"smokybk",class_names_morning.index("fire"):"fire",class_names_morning.index("fireboom"):"fireboom",class_names_morning.index("firesmoky"):"firesmoky"} # 白天事件的類別
EVENT_MORNING_REGULAR = {class_names_morning.index("smokybk"):"smokybk",class_names_morning.index("fire"):"fire",class_names_morning.index("fireboom"):"fire",class_names_morning.index("firesmoky"):"fire"} # 白天事件的類別
EVENT_NIGHT = {class_names_night.index("fire"):"fire",class_names_night.index("fireshine"):"fireshine"} # 晚上事件的類別
EVENT_NIGHT_REGULAR = {class_names_night.index("fire"):"fire",class_names_night.index("fireshine"):"fire"} # 晚上事件的類別

NORMAL_MORNING = {
    "chimney": class_names_morning.index("chimney"),
    "smoky": class_names_morning.index("smoky"),
    "other": class_names_morning.index("other"),
    "windmill": class_names_morning.index("windmill"),
    "tree": class_names_morning.index("tree"),
    "ironframe": class_names_morning.index("ironframe")
}

NORMAL_NIGHT = {
    "chimney": class_names_night.index("chimneydk"),
    "smoky": class_names_night.index("smokydk"),
    "light": class_names_night.index("light"),
    "ironframe": class_names_night.index("ironframe"),
    "moon": class_names_night.index("moon")
}
############### Setting server ###############
SERVICE_PORT = 8080
SERVICE_TOKEN_MODE = False
SERVICE_TOKEN = "test"

############### Setting Line notify ###############
LINEADDRESS = "https://notify-api.line.me/api/notify"
#LINE_TOKEN = "hEmn5VAf5ZtkyFUeVmZs1PDMqECSvrK8WbUNxZ0RW8y" #測試環境
LINE_TOKEN = "dc9zv5AhFDMtLjgPKLFMOt5DfQ5UbxAqWPoQuZTVO71" #正式 #LINE_TOKEN = "kEgmDjqPgkTAQrmdy5GeAcSpd64GbAI2xUu42Xc1Q0A" old version

############### Setting image ###############
IMAGE_STORAGE = "./images/"
EVENT_IMAGE_STORAGE = "event_image"
IMAGE_RESIZE = (1024,576)
KEEP_IMAGE_TIME = 420

############### Setting area ###############
WUFU = "4ab74857-9bed-4f65-afee-0a227e5b7599"
ASIAPOLUMER = "ebabb307-04b1-4cf2-aa61-a609d8978b0f"
FORMOSAPLASTICS = "a8ffb0bd-d0cd-4783-9d1e-785e41675829"
CPC = "e5cd52d4-6364-4df7-9c6f-6b66a9d66bbf"
SHANWEI = "824d5d8c-6319-4f91-a383-8f9c79a89ebc"
NORTH = "e30dbdb2-b392-4fb6-9a2f-06eaf10929ff"
SORTH = "dbe8a60b-98af-444e-8af7-4ea7245d8c0b"
RIGHT = "6eb1f359-c37b-4f0c-a831-51420a0ca19a"
LEFT = "ff27c511-0974-4502-ba07-448f0f9679c7"
MOVE_CPC = "d7ba310f-7158-49ce-a427-133228651b2a"
MOVE_ASIAPOLUMER = "65033b06-d667-45a4-a490-d295cbd14238"
BRIDGE = "fb5df986-fc0c-4d10-b4a4-b8af0c2f9d83"
############### Check rect size###############
NORTH_RECT = [15,5,15,8]
RIGHT_RECT = [-20,10,-20,20]
############### Setting URL###############
GOV_DATA_SUNRISE_SUNSET_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/A-B0062-001?Authorization=rdec-key-123-45678-011121314"

