import cv2
import torch
import torchvision
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

BDCN_DIR = os.path.join(PROJECT_ROOT, "BDCN")
HED_DIR = os.path.join(PROJECT_ROOT, "hed")
SE_MODEL_PATH = os.path.join(PROJECT_ROOT, "se_model", "model.yml")
BDCN_WEIGHT_PATH = os.path.join(
    PROJECT_ROOT,
    "bdcn_model",
    "final-model",
    "bdcn_pretrained_on_bsds500.pth",
)

if BDCN_DIR not in sys.path:
    sys.path.append(BDCN_DIR)
import bdcn

if HED_DIR not in sys.path:
    sys.path.append(HED_DIR)
from hed_network import Network

def get_SE_model():
    if not os.path.exists(SE_MODEL_PATH):
        raise FileNotFoundError(f"SE model file not found: {SE_MODEL_PATH}")
    retval = cv2.ximgproc.createStructuredEdgeDetection(SE_MODEL_PATH)
    return retval

def detect_SE_edge(model, image):
    imgrgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255
    imgrgb = imgrgb.astype(np.float32)
    # model = '/userdir/se_model/model.yml'
    # retval = cv2.ximgproc.createStructuredEdgeDetection(model)
    out = model.detectEdges(imgrgb)
    return out

def get_BDCN_model():
    if not os.path.exists(BDCN_WEIGHT_PATH):
        raise FileNotFoundError(f"BDCN weight file not found: {BDCN_WEIGHT_PATH}")
    bdcn_model = bdcn.BDCN()
    bdcn_model.load_state_dict(torch.load(BDCN_WEIGHT_PATH))
    return bdcn_model

def detect_BDCN_edge(model, image, device):
    data = image.astype(np.float32)
    data -= np.array([[[104.00699, 116.66877, 122.67892]]])
    data = torch.Tensor(data[np.newaxis,:,:,:].transpose(0,3,1,2)).to(device)
    with torch.no_grad():
        out = model(data)
    # fuse = F.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    fuse = torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]
    return fuse

def get_hed_model():
    hed_model = Network()
    return hed_model

def detect_hed_edge(model, image, device):
    raw_height, raw_width = image.shape[0:2]

    image = cv2.resize(image, (480,320))

    tenInput = torch.FloatTensor(np.ascontiguousarray(np.array(image).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # model = model.cuda().eval()
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    assert(intWidth == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert(intHeight == 320) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    tenInput = tenInput.to(device)
    with torch.no_grad():
        tenOutput = model(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :]
    tenOutput = torchvision.transforms.functional.resize(img=tenOutput, size=(raw_height, raw_width))
    return tenOutput.clip(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
