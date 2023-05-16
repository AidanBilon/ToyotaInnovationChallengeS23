import os
HOME = os.getcwd()
print(HOME)

!pip install ultralytics==8.0.20
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="jnx8ZTtnTBQPgVO2tznH")
project = rf.workspace("ml-dataset").project("tmmc-gavyg")
dataset = project.version(3).download("yolov8")

%cd {HOME}
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=20 imgsz=800 plots=True

!ls {HOME}/runs/detect/train/

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/confusion_matrix.png', width=600)

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/results.png', width=600)

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)

%cd {HOME}
!yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True

import glob
from IPython.display import Image, display

for image_path in glob.glob(f'{HOME}/runs/detect/predict3/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")
      
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")

model = project.version(dataset.version).model

"""import os, random
test_set_loc = dataset.location + "/test/images/"
random_test_image = random.choice(os.listdir(test_set_loc))
print("running inference on " + random_test_image)
pred = model.predict(test_set_loc + random_test_image, confidence=40, overlap=30).json()
pred"""

from roboflow import Roboflow
rf = Roboflow(api_key="jnx8ZTtnTBQPgVO2tznH")
project = rf.workspace().project("tmmc-gavyg")
model = project.version(2).model

print(model.predict("https://i.ibb.co/3cHs2R1/IMG-0150.jpg", hosted=True, confidence=40, overlap=30).json())
