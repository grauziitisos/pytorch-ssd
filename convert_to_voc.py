import globox as g
from globox import BoundingBox
import pandas as p
from pathlib import Path
# openimage
#fr = p.read_csv('data\sub-train-annotations-bbox.csv')
fr = p.read_csv('data\sub-test-annotations-bbox.csv')
fr["ImageID"]=[f+'.JPG' for f in fr.ImageID.tolist()] 

fr.to_csv("r.csv", index=False)

openimg = g.AnnotationSet.from_openimage(
    file_path="r.csv",
    #image_folder='data\\train',
    image_folder='data\\test',
    verbose=True
)
openimg.show_stats()

openimg.save_imagenet(save_dir="pas/test/")
a=""
# ImageNet
#yolo.save_openimage(path='r.csv')