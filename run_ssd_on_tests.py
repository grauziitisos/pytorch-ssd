from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.utils.misc import Timer
import cv2
import sys
import torch
import os
from vision.transforms.transforms import *
def run_ssd_on_tests(net_type, model_path, path_type, label_path, image_folder, output_folder):

	class_names = [name.strip() for name in open(label_path).readlines()]
	
	if net_type == 'vgg16-ssd':
		net = create_vgg_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1-ssd':
		net = create_mobilenetv1_ssd(len(class_names), is_test=True)
	elif net_type == 'mb1-ssd-lite':
		net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'mb2-ssd-lite':
		net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'mb3-large-ssd-lite':
		net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'mb3-small-ssd-lite':
		net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
	elif net_type == 'sq-ssd-lite':
		net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
	else:
		print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
		sys.exit(1)
	if(path_type=='1'):
		print("loading state dict...")
		net.load_state_dict(model_path)
	elif(path_type=='2'):
		print("loading parallel state dict...")
		os.environ.setdefault('nproc-per-node','1')
		os.environ.setdefault('max-restarts','9')
		os.environ.setdefault('rdzv-id','3')
		os.environ.setdefault('rdzv-backend','static')
		os.environ.setdefault('rdzv-endpoint','127.0.0.1:7000')
		os.environ.setdefault('nnodes','1')
		os.environ.setdefault('LOCAL_RANK','0')
		os.environ.setdefault('RANK','0')
		os.environ.setdefault('WORLD_SIZE','1')
		os.environ.setdefault('MASTER_ADDR','127.0.0.1')
		os.environ.setdefault('MASTER_PORT','7000')
		torch.distributed.init_process_group(backend="gloo")
		net = torch.nn.parallel.DistributedDataParallel(net,  device_ids=None,
																		output_device=None)
		net.load_state_dict(
						torch.load(model_path))#
	else:
		print("loading net")
		net.load(model_path)
	
	if net_type == 'vgg16-ssd':
		predictor = create_vgg_ssd_predictor(net, candidate_size=200)
	elif net_type == 'mb1-ssd':
		predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
		config = mobilenetv1_ssd_config
	elif net_type == 'mb1-ssd-lite':
		predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
		config = mobilenetv1_ssd_config
	elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
		predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
	elif net_type == 'sq-ssd-lite':
		predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
	else:
		predictor = create_vgg_ssd_predictor(net, candidate_size=200)
	

	for image_path in [ f for f in os.listdir("003trainpush/") if f.endswith(".jpg")]:
		print(image_path)
		exit()
		orig_image = cv2.imread(image_path)
		image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
		#size = config.image_size
		#mean = config.image_mean
		#std = config.image_std
		#transform = Compose([
		#           ToPercentCoords(),
		#           Resize(size),
		#           SubtractMeans(mean),
		#            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
		#            ToTensor(),
		#        ])
		#img, bx, lb = transform(image)
		boxes, labels, probs = predictor.predict(image, 10, 0.25)
		#bobox, lab, pb = predictor.predict(image, 10, 0.3)
		#print(bobox.size(0))
		for i in range(boxes.size(0)):
			box = boxes[i, :]
			print(box)
			cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
			#label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
			label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
			cv2.putText(orig_image, label,
						(int(box[0]) + 20, int(box[1]) + 40),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,  # font scale
						(32, 32, 255),
						2)  # line type
		path = "run_ssd_example_output.jpg"
		cv2.imwrite(path, orig_image)
		print(f"Found {len(probs)} objects. The output image is {path}")


run_ssd_on_tests('sq-ssd-lite', model_path, path_type, label_path, image_folder, output_folder)