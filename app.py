from flask import Flask, render_template, request
from PIL import Image
import cv2
import glob
import json
import os

import pytesseract
import layoutparser as lp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image
app = Flask(__name__)


def ocr_folder(list_img):
	ocr_agent = lp.TesseractAgent.with_tesseract_executable('/usr/bin/tesseract')   
	all_res = []
	for img in list_img:
		image = cv2.imread(img)
		text = pytesseract.image_to_string(img,lang='vie')
		all_res.append(text.replace("\n", "<br>"))
	return all_res

# def pdf_to_image(dir, save_dir):
# 	files_pdf = glob.glob(dir + "/*.pdf")
# 	folder_name = save_dir
# 	if not os.path.exists(folder_name):
# 		os.makedirs(folder_name)
# 	for file in files_pdf:
# 		images = convert_from_path(file)
# 		print(len(images))
# 		for i in range(len(images)):
# 		# Save pages as images in the pdf
# 			#images[i].save(file.split(".")[0] + "_"  + str(i) +'.jpg', 'JPEG')
# 			img_name = file.split("/")[-1]
# 			# images[i] = cv2.resize(images[i],(1654-1,2338-1))
# 			# images[i].save(save_dir + "/" + img_name.split(".")[0] + "_"  + str(i) +'.jpg', 'JPEG')
# 			image = np.array(images[i]) # convert PIL image to NumPy array
# 			image = cv2.resize(image, (1654, 2338))
# 			cv2.imwrite(save_dir + "/" + img_name.split(".")[0] + "_"  + str(i) +'.jpg', image)


def upload_files(folder_name,id_file):
	uploaded_files = request.files.getlist(id_file)
	count_pdf = 0
	for file in uploaded_files:
		filename = file.filename
		if filename.split(".")[-1] == "pdf":
			if not os.path.exists(folder_name + "/pdf"):
				os.makedirs(folder_name + "/pdf")
			file.save(os.path.join(folder_name + "/pdf", filename))
			count_pdf+=1
		else:
			if not os.path.exists(folder_name + "/images"):
				os.makedirs(folder_name + "/images")
			#image = Image.open(file)
			#resized_image = image.resize((new_width, new_height))
			#resized_image.save(os.path.join(folder_name + "/images", filename)) 
			file.save(os.path.join(folder_name + "/images", filename))
	if count_pdf > 0:
		pdf_to_image("static/pdf","static/images")

# def remove_line(crop_img):
# 	# Remove black line
# 	gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
# 	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# 	# Horizontal lines
# 	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
# 	detected_horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# 	# Vertical lines
# 	vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
# 	detected_vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# 	# Combine horizontal and vertical lines
# 	detected_lines = cv2.bitwise_or(detected_horizontal_lines, detected_vertical_lines)
# 	indices = np.where(detected_lines != 0)
# 	crop_img[indices] = (255, 255, 255)
# 	return crop_img

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
	#global image_path, width, height
	global img_cropped
	img_cropped = []
	folder_name = "static/crop_images"
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	#try:
	all_polygons = json.loads(request.form['myList'])
	print("image_path:", image_path)
	for i in range(len(image_path)):
		try:
			for	j in range(len(all_polygons[i])):
				print(all_polygons[i][j])
				x1,x2,y1,y2 = list(map(lambda x: x * ratio, all_polygons[i][j].values()))
				img = cv2.imread(image_path[i])
				img = cv2.resize(img,(width*ratio, height*ratio))
				crop_img = img[y1:y2,x1:x2]
				out_name = folder_name + "/" + image_path[i].split("/")[-1].split(".")[0] + "_" + str(j) + ".jpg"
				img_cropped.append(out_name)
		
				cv2.imwrite(out_name, crop_img)
		except ValueError as e:
			print("Error:", e)

	res = ocr_folder(img_cropped)
	return render_template("result.html",image0 = img_cropped[0], image_path=img_cropped , res0 = res[0], res = res)


@app.route('/', methods=['POST'])
def index2():
    return render_template('index.html')
if __name__ == "__main__":
	app.run(debug=True)