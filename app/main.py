import math
import torch
import cv2
import numpy as np
import shutil
import pandas as pd
import detectron2
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi import Request, Depends,Form
from fastapi import Query
from fastapi.templating import Jinja2Templates
from PIL import Image
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

app = FastAPI()

templates = Jinja2Templates(directory='./app/templates')


#面積算出
def pixcel_count(dete_result):
  area_dict = {}
  pred_class = dete_result["instances"].pred_classes.tolist()
  for idx,types in enumerate(pred_class):
    masks = dete_result['instances'].pred_masks[[idx]]  # マスク情報のテンソルを取得
    indices = torch.nonzero(masks) #マスクから値が0でないものを抽出
    count_rows = torch.sum(indices[:, 0] == 0).item()
    if types in area_dict: #料理とその面積を追加、すでにその料理がある場合は面積を加算
      area_dict[types] += count_rows
    else:
      area_dict[types] = count_rows
  return area_dict


#物体検出
def object_detection(image):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 対応する料理の種類
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "./train/model_1.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #閾値
  cfg.MODEL.DEVICE = "cpu"
  predictor = DefaultPredictor(cfg)
  im = cv2.imread(image)
  dete = predictor(im)
  return pixcel_count(dete)

#カロリー計算
def nutrition_calculation(first_image,second_image):
  first = object_detection(first_image)
  second = object_detection(second_image)
  #全てなくなっている料理の面積は0として追加
  for category, area in first.items(): 
    if category not in second:
        second[category] = 0

  result_dict = {}
  for category in first: #食事前と食事後の面積を比較
    if category in second:
        result_dict[category] = (first[category] - second[category]) / first[category]
  #sorceをもとに料理ごとのカロリーを取得、面積の減少比率から摂取カロリーを算出
  source = './source/class.csv'
  df = pd.read_csv(source)
  class_name = df['name']
  calorie = df['calorie']
  categories = {}
  all_calorie = 0
  for class_id in result_dict:
    if class_id != 1: #class_id:1は皿でカロリーがないため無視
      calorie_intake = round(calorie.iloc[class_id] * result_dict[class_id],1)
      intake_category = class_name.iloc[class_id]
      print(f'{intake_category}の摂取カロリー:{calorie_intake}kcal')
      categories[intake_category] = str(calorie_intake) + 'kcal'
      all_calorie += calorie_intake
  return str(categories).strip('{}'),f'総摂取カロリー{all_calorie}kcal'

#入力画像の保存
def get_uploadfile(upload_file: UploadFile):
    path = f'./image/{upload_file.filename}'
    with open(path, 'wb+') as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return {
        'filename': path,
        'type': upload_file.content_type
    }

@app.get( "/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse('page.html', {"request": request})


@app.post("/upload")
async def process_images(request: Request,image1: UploadFile = File(...), image2: UploadFile = File(...),result: float = Form(0.0)):
    # アップロードされた画像を読み込む

  first_image = get_uploadfile(image1)
  second_image = get_uploadfile(image2)
  first_path = first_image['filename']
  second_path = second_image['filename']
  result = str(nutrition_calculation(first_path,second_path)).replace('"', '').replace("(","").replace(")","").replace("'","")
  result_html = f'<p>{result}</p>'
  os.remove(first_path)
  os.remove(second_path)

  return templates.TemplateResponse(
        'page.html',
        {"request": request, "result_html": result_html}
    )
