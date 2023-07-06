from flask import Flask, render_template, request
from PIL import Image
import cv2
import glob
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image
app = Flask(__name__)



@app.route('/', methods=['Get'])
def index():
    return render_template('index.html')

@app.route("/label", methods=['POST'])
def home():
	img_files = request.files.getlist("img_files")
	for file in img_files:
		filename = file.filename
		if not os.path.exists("static/images"):
				os.makedirs("static/images")
		file.save(os.path.join("static/images", filename))

	if not os.path.exists('static/labels'):
		os.makedirs('static/labels')
	#cmd = 'python yolov7/detect.py --weights yolov7/best_e6e.pt --source static/images --save-txt --device cpu'
	#os.system(cmd)
	global width, height, image_path,ratio
	ratio = 2	#Ti le thu nho anh hien thi len Web
	image_path = glob.glob("static/images/*.jpg") + glob.glob("static/images/*.jpeg") + glob.glob("static/images/*.png")
	img = cv2.imread(image_path[0])
	h, w,_ = img.shape
	width = int(w/ratio)
	height = int(h/ratio)
	txt_path = glob.glob('static/labels/*.txt')

	all_polygons = [] #[[{'x1': 96, 'y1': 149, 'x2': 403, 'y2': 228}],[{'x1': 96, 'y1': 149, 'x2': 403, 'y2': 300}]]
	for txt in txt_path:
		polygon = []
		with open(txt, 'r') as f:
			lines = f.read().strip().split('\n')
		for line in lines:
			text,x1,y1,x2,y2 =  list(map(lambda x: int(int(x) / ratio), line.split(' '))) #line.split(' ')
			rectangles = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,'text':text}
			#print(list(map(lambda x: x * ratio, rectangles.values())))
			polygon.append(rectangles)
		all_polygons.append(polygon)
	return render_template("label.html",image0=image_path[0], image_path=image_path, width=width,height=height,all_polygons=all_polygons)


@app.route('/result', methods=['POST'])
def crop_detect():
	return render_template("result.html")


@app.route('/', methods=['POST'])
def index2():
    return render_template('index.html')
if __name__ == "__main__":
	app.run(debug=True)