import numpy as np
import torch
import pdb
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def IOU_matrix(box):
  
    iou_matrix = torch.zeros((box[0].shape[0],box[0].shape[0]))
    for i in range(box[0].shape[0]):
        for j in range(box[0].shape[0]):
            if i==j:
                iou_matrix[i][j] = 1
            else:
                xA = max(box[0][i], box[0][j])
                yA = max(box[1][i], box[1][j])
                xB = min(box[2][i], box[2][j])
                yB = min(box[3][i], box[3][j])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (box[2][i] - box[0][i] + 1) * (box[3][i] - box[1][i] + 1)
                boxBArea = (box[2][j] - box[0][j] + 1) * (box[3][j] - box[1][j] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou_matrix[i,j] = interArea / float(boxAArea + boxBArea - interArea)
                # return the intersection over union value
    return iou_matrix

def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    #interArea = (xB - xA) *(yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    # flatten_regr = torch.view(-1,4)
    # flatten_clas = torch.view(-1)
    # flatten_anchors = torch.view(-1,4)
    if len(out_r.shape)==3:
      bz = 1
    elif len(out_r.shape)==4:
      bz = out_r.shape[0]
    grid_size_0 = 50
    grid_size_1 = 68
    out_r_permuted = out_r.permute(0,2,3,1)
    print(out_r_permuted.shape)
    flatten_regr = out_r.reshape((bz*grid_size_0*grid_size_1,4))
    flatten_clas = torch.flatten(out_c)
    flatten_anchors = anchors.expand(bz,-1,-1,-1).reshape((bz*grid_size_0*grid_size_1,4))

    return flatten_regr.cuda(), flatten_clas.cuda(), flatten_anchors.cuda()




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the output
    #######################################

    box = torch.zeros(flatten_out.shape).cuda()
    # box[:,0:2] = (flatten_out[:,0:2] * flatten_anchors[:,2:4]) + flatten_anchors[:,0:2]
    # box[:,2:4] = torch.exp(flatten_out[:,2:4]) * flatten_anchors[:,2:4]
    for i in range(len(flatten_out)):
      box[i,0] = (flatten_out[i,0] * flatten_anchors[i,2]) + flatten_anchors[i,0]
      box[i,1] = (flatten_out[i,1] * flatten_anchors[i,3]) + flatten_anchors[i,1]
      box[i,2] = torch.exp(flatten_out[i,2]) * flatten_anchors[i,2]
      box[i,3] = torch.exp(flatten_out[i,3]) * flatten_anchors[i,3]
      # print(box[i,:])
    box_corners = torch.zeros(flatten_out.shape)
    box_corners[:,0] = box[:,0] - box[:,2]/2
    box_corners[:,1] = box[:,1] - box[:,3]/2
    box_corners[:,2] = box[:,0] + box[:,2]/2
    box_corners[:,3] = box[:,1] + box[:,3]/2
    # flatten_out_shape = flatten_out.shape[0]
    return box,box_corners