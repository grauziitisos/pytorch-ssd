# pip install -U label-studio
import argparse
import os
import logging
import sys
import itertools
from datetime import datetime

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from datetime import *
from torch.utils.data.distributed import DistributedSampler
from pprint import pformat as pf
import pickle

__DEBUG1__= False #True
#restora
__DEBUG2__=False#True
# OK
#__Deb_2Str__="models/2023-11-04_18-47-55_init_mb1-ssd.pth"
__Deb_2Str__="models/op/2023-11-02_20-57-51mb1-ssd-Epoch-9-.weights"
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
                    
parser.add_argument('--in_mod', default=1, type=int,
                    help='normal or parallel. 0 normal 1 parallel dict')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--node_rank", default=0, type=int)
parser.add_argument("--nproc_per_node", default=1, type=int)
parser.add_argument("--world_size", default=5, type=int)
parser.add_argument("--master_addr", default='127.0.0.1', type=str)
parser.add_argument("--master_port", default=7000, type=int)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
args = parser.parse_args()
opt = args
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

#os.environ["MASTER_ADDR"] = str(opt.master_addr)
#os.environ["MASTER_PORT"] = str(opt.master_port)

def pickle_trick(obj, max_depth=10):
    output = {}

    if max_depth <= 0:
        return output

    try:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)
        print(obj)
        print(failing_children)
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        failing_children = []

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj, 
            "err": e, 
            "depth": max_depth, 
            "failing_children": failing_children
        }

    return output

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def test(loader, net, criterion, device, net_type='sq-ssd-lite'):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num, regression_loss, classification_loss 
    
def draw_tests(net, device, net_type='sq-ssd-lite'):
    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    
    
    timer = Timer()
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    
            cv2.putText(orig_image, label,
                        (int(box[0])+20, int(box[1])+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
    


if __name__ == '__main__':
    timer = Timer()
    if(__DEBUG1__):
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
###############################################################################
        # parser.__setattr__("--dataset_type","open_images")
        # parser.add_argument("--dataset_type", type=str, default="open_images", help="for debug purposes..")
        # parser.__setattr__("--datasets","data")
        # parser.add_argument("--datasets", type=str, default="data", help="for debug purposes..")
        args.__setattr__("dataset_type", "open_images")
        args.__setattr__("datasets", ["data"])
        args.__setattr__("net", "mb1-ssd")
        args.__setattr__("pretrained_ssd", "models/mobilenet-v1-ssd-mp-0_675.pth")
        # ??DEVUG2==
        if(__DEBUG2__):
            args.__setattr__("resume", __Deb_2Str__)
        args.__setattr__("scheduler", "cosine")
        args.__setattr__("lr", float("0.01"))
        args.__setattr__("t_max", int("100"))
        args.__setattr__("validation_epochs", int("5"))
        args.__setattr__("num_epochs", int("20"))
        args.__setattr__("base_net_lr", float("0.001"))
        args.__setattr__("batch_size", int("5"))
        args.__setattr__("num_workers", int("1"))
    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        create_net = lambda num: create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        create_net = lambda num: create_mobilenetv3_small_ssd_lite(num)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    if os.environ.get('LOCAL_RANK'):
        print(str(os.environ['LOCAL_RANK']))
    print(f'net name: {args.net}')
#    if(str(os.environ['LOCAL_RANK'])==""):
#        os.environ['LOCAL_RANK'] = opt.node_rank
#    else:
#        opt.node_rank = os.environ['LOCAL_RANK']
 #   torch.distributed.init_process_group(backend='gloo', rank=opt.node_rank, world_size=opt.world_size, timeout=timedelta(seconds=3600))
    torch.distributed.init_process_group(backend="gloo")
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    len_images_prev = 0
    len_images_now = 0
    first_launch = True

    last_epoch = -1
    #unimplemented parameter???
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        logging.info("Checking for train dataset updates...")
        len_images_prev = len_images_now
        len_dsets = 0
        for dataset_path in args.datasets:
            if args.dataset_type == 'voc':
                dataset = VOCDataset(dataset_path, transform=train_transform,
                                     target_transform=target_transform)
                label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
                store_labels(label_file, dataset.class_names)
                num_classes = len(dataset.class_names)
            elif args.dataset_type == 'open_images':
                dataset = OpenImagesDataset(dataset_path,
                     transform=train_transform, target_transform=target_transform,
                     dataset_type="train", balance_data=args.balance_data)
                label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
                store_labels(label_file, dataset.class_names)
                logging.info(dataset)
                num_classes = len(dataset.class_names)
            else:
                raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
            len_dsets = len_dsets + len(dataset)
            del dataset
        len_images_now = len_dsets
        if(len_images_now != len_images_prev):
            len_images_prev = len_images_now
            logging.info("Initializing datasets. First run or new images appeared compared to previous iteration in this launch.")
            logging.info("Prepare training datasets.")
            datasets = []
            len_dsets = 0
            for dataset_path in args.datasets:
                if args.dataset_type == 'voc':
                    dataset = VOCDataset(dataset_path, transform=train_transform,
                                         target_transform=target_transform)
                    label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
                    store_labels(label_file, dataset.class_names)
                    num_classes = len(dataset.class_names)
                elif args.dataset_type == 'open_images':
                    dataset = OpenImagesDataset(dataset_path,
                         transform=train_transform, target_transform=target_transform,
                         dataset_type="train", balance_data=args.balance_data)
                    label_file = os.path.join(args.checkpoint_folder, "open-images-model-labels.txt")
                    store_labels(label_file, dataset.class_names)
                    logging.info(dataset)
                    num_classes = len(dataset.class_names)

                else:
                    raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
                len_dsets = len_dsets + len(dataset)
                datasets.append(dataset)
            logging.info(f"Stored labels into file {label_file}.")
            train_dataset = ConcatDataset(datasets)
            len_images_now = len_dsets
            del datasets
            logging.info("Train dataset size: {}".format(len(train_dataset)))
            train_loader = DataLoader(train_dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)
            #todo: add distributed devices !!!!!!!!!!!!!!!!!!!!!!!
            logging.info("distributed devices....")
            dist_sampler = DistributedSampler(dataset)
            train_loader = DataLoader(dataset, sampler=dist_sampler, num_workers=opt.n_cpu)#, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

           # for i, data in enumerate(train_loader):
           #     images, boxes, labels = data
           #     print("----------------")
           #     print(len(images[0]))
           #     print(boxes[0])
           #     print("--------------111111111111")


            logging.info("Prepare Validation datasets.")
            if args.dataset_type == "voc":
                val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                         target_transform=target_transform, is_test=True)
            elif args.dataset_type == 'open_images':
                val_dataset = OpenImagesDataset(dataset_path,
                                                transform=test_transform, target_transform=target_transform,
                                                dataset_type="test")
                logging.info(val_dataset)
            logging.info("validation dataset size: {}".format(len(val_dataset)))

            val_loader = DataLoader(val_dataset, args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)
            cuda = args.use_cuda and torch.cuda.is_available()
            #Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
            if(first_launch):
                logging.info("Build network.")
                net = create_net(num_classes)
                net_best = create_net(num_classes)
                min_loss = -10000.0

                base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
                extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
                if args.freeze_base_net:
                    logging.info("Freeze base net.")
                    freeze_net_layers(net.base_net)
                    params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                             net.regression_headers.parameters(), net.classification_headers.parameters())
                    params = [
                        {'params': itertools.chain(
                            net.source_layer_add_ons.parameters(),
                            net.extras.parameters()
                        ), 'lr': extra_layers_lr},
                        {'params': itertools.chain(
                            net.regression_headers.parameters(),
                            net.classification_headers.parameters()
                        )}
                    ]
                elif args.freeze_net:
                    freeze_net_layers(net.base_net)
                    freeze_net_layers(net.source_layer_add_ons)
                    freeze_net_layers(net.extras)
                    params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
                    logging.info("Freeze all the layers except prediction heads.")
                else:
                    params = [
                        {'params': net.base_net.parameters(), 'lr': base_net_lr},
                        {'params': itertools.chain(
                            net.source_layer_add_ons.parameters(),
                            net.extras.parameters()
                        ), 'lr': extra_layers_lr},
                        {'params': itertools.chain(
                            net.regression_headers.parameters(),
                            net.classification_headers.parameters()
                        )}
                    ]

                is_HAL_here = False
                timer.start("Load Model")
                if args.resume:
                    logging.info(f"Resume from the model {args.resume}")
                    if args.pretrained_ssd:
                       logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
                       net.init_from_pretrained_ssd(args.pretrained_ssd)
                       net_best.init_from_pretrained_ssd(args.pretrained_ssd)
                    elif args.base_net:
                       logging.info(f"Init from base net {args.base_net}")
                       net.init_from_base_net(args.base_net)
                       net_best.init_from_base_net(args.base_net)

                       #net.load(args.resume)
                       #print("loaeed?")
#2023-12-31: why these 2 were here? how long??
#bc no need to resume, that is why
                    elif(args.net=='sq-ssd-lite'):
                       a=""
                    else:
                       net.load(args.resume)
                    if args.pretrained_ssd or args.base_net or args.net=='sq-ssd-lite':
                        if(args.in_mod==1):
                            #print(f"in_mod: {args.in_mod}")
                            net = torch.nn.parallel.DistributedDataParallel(net,  device_ids=None,
                                                                      output_device=None)
                            net_best = torch.nn.parallel.DistributedDataParallel(net_best,  device_ids=None,
                                                                      output_device=None)

                            net.load_state_dict(
                            torch.load(args.resume))#, map_location=map_location))
                            net_best.load_state_dict(
                            torch.load(args.resume))#, map_location=map_location))

                        else:
                            #print(f"nooo in_mod: {args.in_mod}")
                            net.load_state_dict(
                            torch.load(args.resume))#, map_location=map_location))
                            net_best.load_state_dict(
                            torch.load(args.resume))#, map_location=map_location))

                            net = torch.nn.parallel.DistributedDataParallel(net,  device_ids=None,
                                                                      output_device=None)
                            net_best = torch.nn.parallel.DistributedDataParallel(net_best,  device_ids=None,
                                                                      output_device=None)

                            # nope, no help
                            # torch.distributed.barrier()
                    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
                    is_HAL_here = True
                elif args.base_net:
                    logging.info(f"Init from base net {args.base_net}")
                    net.init_from_base_net(args.base_net)
                    net_best.init_from_base_net(args.base_net)

                elif args.pretrained_ssd:
                    logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
                    net.init_from_pretrained_ssd(args.pretrained_ssd)
                    net_best.init_from_pretrained_ssd(args.pretrained_ssd)
                logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
                timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # print(pf(pickle_trick(net.state_dict())))
                #torch.save(net.state_dict(), os.path.join(args.checkpoint_folder, f"{timestr}_init_{args.net}.pth"))
                #net.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, f"{timestr}_init_{args.net}.pth")))
                #barenet=net
                net_best.to(DEVICE)
                net.to(DEVICE)
                local_rank = int(os.environ["LOCAL_RANK"])
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(local_rank)
                print(os.environ["RANK"])
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if not is_HAL_here:
                    net = torch.nn.parallel.DistributedDataParallel(net,  device_ids=None,
                                                                      output_device=None)

                criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                                         center_variance=0.1, size_variance=0.2, device=DEVICE)
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                             + f"Extra Layers learning rate: {extra_layers_lr}.")

                if args.scheduler == 'multi-step':
                    logging.info("Uses MultiStepLR scheduler.")
                    milestones = [int(v.strip()) for v in args.milestones.split(",")]
                    scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                                 gamma=0.1, last_epoch=last_epoch)
                elif args.scheduler == 'cosine':
                    logging.info("Uses CosineAnnealingLR scheduler.")
                    scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
                else:
                    logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
                    parser.print_help(sys.stderr)
                    sys.exit(1)
                first_launch = False
            
            #scheduler.step()
            #print('bare net backup before train....')
            #torch.save(net.state_dict(), os.path.join(args.checkpoint_folder, f"{timestr}_init_{args.net}.pth"))
            #torch.distributed.barrier()
            #net.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, f"{timestr}_init_{args.net}.pth")))

            #barenet.save('test.pth')
            #torch.distributed.barrier()
            #barenet.load('test.pth')
            cantformatthis = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            val_loss_best, val_regression_loss_best, val_classification_loss_best, val_locate_best, val_confclass_best = test(val_loader, net_best, criterion, DEVICE)
            logging.info(
                f"test best: {cantformatthis}, " +
                f"Validation Loss: {val_loss_best:.4f}, " +
                f"Validation Regression Loss {val_regression_loss_best:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss_best:.4f}"
            )

        # draw_tests(net, DEVICE, net_type='sq-ssd-lite')
        #torch.save(net.module.state_dict(), "t.n")
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()
        torch.distributed.barrier()
        timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(os.path.join(args.checkpoint_folder, f"{timestr}{args.net}-Epoch-{epoch}-.weights"))
        model_path=os.path.join(args.checkpoint_folder, f"{timestr}{args.net}-Epoch-{epoch}-.weights")
        torch.save(net.module.state_dict(), model_path)
        print("barrier...")
        torch.distributed.barrier()
        #map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        net.module.load_state_dict(
        torch.load(model_path))#, map_location=map_location))
        logging.info(f"Saved an re-loaded inner model state {model_path}")
        # draw_tests(net, DEVICE, net_type='sq-ssd-lite')
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss, val_locate, val_confclass = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_path = os.path.join(args.checkpoint_folder, f"{timestr}{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            torch.save(net.module.state_dict(), model_path)
            logging.info(f"Saved inner model state {model_path}")

            if(val_locate < val_locate_best or val_confclass < val_confclass_best):
            	logging.info("model improved. saving best")
            	torch.save(net.module.state_dict(), os.path.join(args.checkpoint_folder, f"{timestr}_model_best_{args.net}-e-{epoch}.pth"))
            	net_best.module.load_state_dict(torch.load(os.path.join(args.checkpoint_folder, f"{timestr}_model_best_{args.net}-e-{epoch}.pth")))
            	val_locate_best = val_locate
            	val_confclass_best = val_confclass
            else:
            	logging.info("failed attempt to improve model. reseting to last best state dict")
            	torch.save(net_best.module.state_dict(), "temp.pth")
            	net.module.load_state_dict(torch.load("temp.pth"))

            #print("barrier.wwww..")
            #torch.distributed.barrier()
            #map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            #net.load_state_dict(
            #torch.load(model_path))#, map_location=map_location))
    timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(args.checkpoint_folder, f"fi_dis_{timestr}_{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
    torch.save(net.state_dict(), model_path)
    logging.info(f"Saved final dis model state {model_path}")
    model_path = os.path.join(args.checkpoint_folder, f"_fi__{timestr}_{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
    torch.save(net.module.state_dict(), model_path)
    logging.info(f"Saved final model state {model_path}")

	
