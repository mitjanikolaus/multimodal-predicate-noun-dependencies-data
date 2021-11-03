# !/usr/bin/env python
# Script adapted from https://github.com/airsplay/lxmert

# The root of bottom-up-attention repo. Do not need to change if using provided docker file.
BUTD_ROOT = '/opt/butd/'

import os, sys
sys.path.insert(0, BUTD_ROOT + "/tools")
os.environ['GLOG_minloglevel'] = '2'

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

import caffe
import argparse
import pprint
import base64
import numpy as np
import cv2
import csv
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 36
MAX_BOXES = 36


def generate_tsv(prototxt, weights, image_paths, outfile):
    # First check if file exists, and if it is complete
    # never use set, it loses the order!!! F***
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(item['img_id'])

    caffe.set_mode_cpu()
    # caffe.set_device(0)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for path in tqdm(image_paths):
            print(path)
            try:
                writer.writerow(get_detections_from_im(net, path))
            except Exception as e:
                print(e)


def get_detections_from_im(net, path, conf_thresh=0.2):
    """
    :param net:
    :param path: full path to an image
    :param conf_thresh:
    :return: all information from detection and attr prediction
    """
    im = cv2.imread(path)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    objects_conf = np.max(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)
    attrs_conf = np.max(attr_prob[keep_boxes][:, 1:], axis=1)

    return {
        "img_id": path,
        "img_h": np.size(im, 0),
        "img_w": np.size(im, 1),
        "objects_id": base64.b64encode(objects),  # int64
        "objects_conf": base64.b64encode(objects_conf),  # float32
        "attrs_id": base64.b64encode(attrs),  # int64
        "attrs_conf": base64.b64encode(attrs_conf),  # float32
        "num_boxes": len(keep_boxes),
        "boxes": base64.b64encode(cls_boxes[keep_boxes]),  # float32
        "features": base64.b64encode(pool5[keep_boxes])  # float32
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=BUTD_ROOT + "models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt", type=str)
    parser.add_argument('--outfile', dest='outfile',
                        help='output filepath',
                        required=True, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=BUTD_ROOT + "experiments/cfgs/faster_rcnn_end2end_resnet.yml", type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--imgroot', type=str, default='/workspace/images/')
    parser.add_argument('--img-paths', type=str, required=True)
    parser.add_argument('--caffemodel', type=str, default='./resnet101_faster_rcnn_final_iter_320000.caffemodel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup the configuration, normally do not need to touch these:
    args = parse_args()


    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    t = open(args.img_paths, "r")
    paths = t.readlines()
    t.close()

    docker_paths = []
    for path in paths:
        path = path.replace("\n", "")
        # for normal images:
        if "fiftyone/" in path:
            path = os.path.join("/workspace/images/", path.split("fiftyone/open-images-v6/")[1])
        # for cropped images:
        elif "images_cropped/" in path:
            path = os.path.join("/workspace/images/", path.split("images_cropped/")[1])
        else:
            raise RuntimeError("Unsupported Path: ", path)

        docker_paths.append(path)

    generate_tsv(args.prototxt, args.caffemodel, docker_paths, args.outfile)
