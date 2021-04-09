import cv2
import subprocess, os, platform,time
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog



# getting File
class GetFileName():
	base = None
	res = ""
	def __init__(self):
		self.base = Tk()
		# Create a canvas
		self.base.geometry('150x150')
		# Button label
		x = Button(self.base, text ='Select a image file', command = self.file_opener)
		x.place(relx=0.25,rely=0.25)
		# Setting Title
		self.base.title("WeirdLens")
		# Setting Size
		self.base.geometry("300x300")
		self.base.mainloop()


	def file_opener(self):
		# Open File Dialog
		result = filedialog.askopenfilename(initialdir="./",title="WeirdLens",filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif',
                                                       '.tiff', '.tif', '.bmp'])])
		# Destroy the Base After Getting File
		self.base.destroy()
		self.base = None
		# Setting The Global result
		self.res = result

# Detecting Object
class WeirdLens():

	def __init__(self):
		thres = 0.6 # Threshold to detect object

		getfilename = GetFileName() # initialising GetFileName Class
		img_file = getfilename.res # Getting FileName or Address
		getfilename.base = None; # Destroying Window
		img = cv2.imread(img_file) # Reading Image File

		classNames= []
		classFile = 'coco.names'
		with open(classFile,'rt') as f:
		    classNames = f.read().rstrip('\n').split('\n')

		# All Models
		configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
		weightsPath = 'frozen_inference_graph.pb'

		net = cv2.dnn_DetectionModel(weightsPath,configPath)
		net.setInputSize(320,320)
		net.setInputScale(1.0/ 127.5)
		net.setInputMean((127.5, 127.5, 127.5))
		net.setInputSwapRB(True)

		# Detecting Objects
		classIds, confs, bbox = net.detect(img,confThreshold=thres)

		# Objects Detected Variables
		self.objectsDetected = []


		# Drawing Box around Image
		if len(classIds) != 0:
			for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
				self.objectsDetected.append(classNames[classId-1])
				cv2.rectangle(img,box,color=(0,255,0),thickness=2)
				cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
				            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

		print(self.objectsDetected)

		# Saving The Image as Temporary File
		cv2.imwrite("output.png",img)
		self.displayImage(img.shape)

	def displayImage(self,dimensions):
		if platform.system() == 'Darwin':       # macOS
			subprocess.call(('open', "output.png"))
			time.sleep(0.5)
			subprocess.call(("rm","output.png"))
		elif platform.system() == 'Windows':    # Windows
			os.startfile("output.png")
			time.sleep(0.5)
			subprocess.call(("del","output.png"))
		else:                                   # linux variants
			subprocess.call(('xdg-open', "output.ong"))
			time.sleep(0.5)
			subprocess.call(("rm","output.png"))


if __name__ == "__main__":
	WeirdLens()