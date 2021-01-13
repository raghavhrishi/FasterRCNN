import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import pdb
import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride = 1, padding=2),     # layer 1
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride = 1, padding=2),    # layer 2
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding=2),    # layer 3
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride = 1, padding=2),   # layer 4
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride = 1, padding=2),  # layer 5
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        # TODO  Define Intermediate Layer
        self.intermediate_layer = torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)

        # TODO  Define Proposal Classifier Head
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            torch.nn.Sigmoid()
        )

        # TODO Define Proposal Regressor Head
        self.regressor = torch.nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1)

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        backbone_out = self.forward_backbone(X)

        #TODO forward through the Intermediate layer
        out_intermediate = self.intermediate_layer(backbone_out)

        #TODO forward through the Classifier Head
        logits = self.classifier(out_intermediate)

        #TODO forward through the Regressor Head
        bbox_regs = self.regressor(out_intermediate)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        #####################################
        X = self.backbone(X)

        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        anchors = torch.zeros((grid_sizes[0] , grid_sizes[1],4))
        for i in range(anchors.shape[0]):
            for j in range(anchors.shape[1]):
                anchors[i][j][0] = stride*j + (stride/2)
                anchors[i][j][1] = stride*i + (stride/2)
                anchors[i][j][2] = (aspect_ratio*scale**2)**0.5
                anchors[i][j][3] = ((scale)**2/aspect_ratio)**0.5
                
        assert anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

        return anchors



    def get_anchors(self):
        return self.anchors

    def get_grid_coordinates(self, x_iter, y_iter, grid_h, grid_w):
      grid_top_left_x = grid_w*x_iter
      grid_top_left_y = grid_h*y_iter
      grid_bottom_right_x = grid_top_left_x + grid_w
      grid_bottom_right_y = grid_top_left_y + grid_h
      return [grid_top_left_x, grid_top_left_y, grid_bottom_right_x, grid_bottom_right_y]

    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        ground_clas_list = []
        ground_coord_list = []
        for i in range(2):
          ground_clas_single, ground_coord_single = self.create_ground_truth(bboxes_list[i], indexes[i], [50,68], self.get_anchors(), [800,1088])
          ground_clas_list.append(ground_clas_single)
          ground_coord_list.append(ground_coord_single)
          # plt.imshow(ground_clas_single[0,:,:])
          # plt.show()
        # ground_clas, ground_coord = MultiApply(self.create_ground_truth,bboxes_list,indexes,list(self.anchors_param['grid_size']),self.get_anchors(),image_shape)
        ground_clas = torch.stack(ground_clas_list,dim=0)
        ground_coord = torch.stack(ground_coord_list,dim=0)
        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        return ground_clas, ground_coord


    def encode(self,anchor_box,ground_truth):
      tx = (ground_truth[0] - anchor_box[0])/anchor_box[2]
      ty = (ground_truth[1] - anchor_box[1])/anchor_box[3]
      tw = torch.log(ground_truth[2]/anchor_box[2])
      th = torch.log(ground_truth[3]/anchor_box[3])
      return [tx,ty,tw,th]

    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord 
        # print("Hello")      
        ground_clas = torch.ones((1,grid_size[0],grid_size[1])) * -1
        ground_coord = torch.zeros((4,grid_size[0],grid_size[1]))
        max_iou = torch.zeros((grid_size[0],grid_size[1]))
        for idx,bbox1 in enumerate(bboxes):
          # print("Bounding Box",bboxes)  
          # print("Bounding Box",bbox1)  
          #bbox1 = list(itertools.chain(*bbox1))
          bbox_w = bbox1[2] - bbox1[0]
          bbox_h = bbox1[3] - bbox1[1]
          #bbox_center_x =  ((bbox1[2] + bbox1[0])/2)
          bbox_center_x = (bbox1[0] + bbox_w/2).item()
          bbox_center_y = (bbox1[1] + bbox_h/2).item()
         # bbox_center_y =  ((bbox1[3] + bbox1[1])/2)
          bbox_list = [bbox_center_x,bbox_center_y,bbox_w,bbox_h] 
          
          maximum_iou = -1    
          maximum_i = []
          maximum_j = []   
          for i in range(grid_size[0]):
            for j in range(grid_size[1]):
              anchor_val = anchors[i][j]
              anchor_center_x = anchor_val[0]
              anchor_center_y = anchor_val[1]
              anchor_w = anchor_val[2]
              anchor_h = anchor_val[3]
              anchor_x1 = (anchor_center_x - anchor_w/2).item()
              anchor_x2 = (anchor_center_x + anchor_w/2).item()
              anchor_y1 = (anchor_center_y - anchor_h/2).item()
              anchor_y2 = (anchor_center_y + anchor_h/2).item()
              anchor_list = [anchor_x1,anchor_y1,anchor_x2,anchor_y2]
              # print("BBOX List",bbox1)
              # print("Anchor List",anchor_list)
              #Cross Boundary Anchors 
              if (anchor_x1 < 0 or anchor_x2 > 1088 or anchor_y1 < 0 or anchor_y2 > 800):
                continue
              #IOU Calculation 
              IOU_comp = IOU(bbox1,anchor_list)
              if IOU_comp > max_iou[i][j]:
                max_iou[i][j] = IOU_comp

              # print("IOU value",IOU_comp)
              if IOU_comp > maximum_iou *1.01:
                maximum_iou = IOU_comp
                maximum_i = [i]
                maximum_j = [j]
                anchor_box_list = [anchor_val]
              elif IOU_comp ==  maximum_iou or IOU_comp > 0.99 * maximum_iou :
                maximum_iou = max(IOU_comp,maximum_iou)
                maximum_i.append(i)
                maximum_j.append(j)
                anchor_box_list.append(anchor_val)
              if IOU_comp > 0.7:
                ground_clas[0][i][j] = 1
                list_encode = self.encode(anchor_val,bbox_list)
                ground_coord[0][i][j] = list_encode[0]
                ground_coord[1][i][j] = list_encode[1]
                ground_coord[2][i][j] = list_encode[2]
                ground_coord[3][i][j] = list_encode[3]
          
                  
          for i,j,anchor_box in zip(maximum_i,maximum_j,anchor_box_list):
            #print("i","j","anchor_box",i,j,anchor_box)
            ground_clas[0,i,j] = 1
            list_encode = self.encode(anchor_box,bbox_list)
            ground_coord[0][i][j] = list_encode[0]
            ground_coord[1][i][j] = list_encode[1]
            ground_coord[2][i][j] = list_encode[2]
            ground_coord[3][i][j] = list_encode[3]
          
          
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
              anchor_val = anchors[i][j]
              anchor_center_x = anchor_val[0]
              anchor_center_y = anchor_val[1]
              anchor_w = anchor_val[2]
              anchor_h = anchor_val[3]
              anchor_x1 = anchor_center_x - anchor_w/2
              anchor_x2 = anchor_center_x + anchor_w/2
              anchor_y1 = anchor_center_y - anchor_h/2
              anchor_y2 = anchor_center_y + anchor_h/2
              anchor_list = [anchor_x1,anchor_y1,anchor_x2,anchor_y2]
              # print(anchor_x1,anchor_y1,anchor_x2,anchor_y2)
              if (anchor_x1 < 0 or anchor_x2 > 1088 or anchor_y1 < 0 or anchor_y2 > 800):
                continue
              # print("noooooooooooooooooooo")
              # if max_iou[i][j]<0.3:
              #   print("yes")
              mat = ground_clas[0,i,j]==1
              if (max_iou[i][j] <  0.3 and not mat):
                # print("Negative")
                ground_clas[0,i,j] = 0
              

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        self.ground_dict[key] = (ground_clas, ground_coord)
        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])
        return ground_clas, ground_coord





    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):
        BCE_loss = torch.nn.BCELoss()
        loss = BCE_loss(p_out,n_out)
        # sum_count = (torch.ones(p_out) + torch.ones(n_out))/2
        return loss



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
        reg_loss = torch.nn.SmoothL1Loss()
        loss = reg_loss(pos_target_coord,pos_out_r)
        return loss



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=1, effective_batch=50):
        #Total Length = M
        clas_out = clas_out.cuda()
        regr_out = regr_out.cuda()
        target_obj = target_obj.cuda()
        targ_regr = targ_regr.cuda()
        M = effective_batch*target_obj.size(0)
        #Flattening the inputs
        target_flatten = target_obj.view(-1)
        clas_out_flatten = clas_out.view(-1)
        #Regressor
        #4x50x68x4 and then flatten
        targ_regr_flatten = targ_regr.permute(0,2,3,1).reshape(-1,4)
        #print("Target",targ_regr_flatten.shape)
        regr_out_flatten = regr_out.permute(0,2,3,1).reshape(-1,4)
        #print("Actual",regr_out_flatten.shape)
        #Subsampling of Positives and Negatives
        positive_indices = (target_flatten == 1).nonzero()
        negative_indices = (target_flatten == 0).nonzero()
        positive_indices = positive_indices[torch.randperm(positive_indices.size()[0])]
        negative_indices = negative_indices[torch.randperm(negative_indices.size()[0])]
        correct_pos_ind = 0
        correct_neg_ind = 0
        #print("Positive Indices",type(positive_indices))
        #Appending the positive indices if not equal to half
        if positive_indices.size()[0] > int(M*0.5):
            correct_pos_ind = positive_indices[:int(M*0.5)]
            correct_neg_ind = negative_indices[:M-int(M*0.5)]
        else:
            correct_pos_ind = positive_indices
            correct_neg_ind = negative_indices[:M-positive_indices.size()[0]]
        indices = torch.cat((correct_pos_ind,correct_neg_ind),0)
        subsample_target_obj = target_flatten[indices]
        subsample_clas_out = clas_out_flatten[indices]
        #Squeezing coz 160x1x4 t0 160x4
        subsample_targ_regr = targ_regr_flatten[indices].squeeze(1)
        subsample_regr_out = regr_out_flatten[indices].squeeze(1)
        # print("subsample_regr_out",subsample_regr_out.shape)
        # print("subsample_targ_regr",subsample_targ_regr.shape)
        #Classifier Loss 
        classifier_loss = self.loss_class(subsample_clas_out.float(),subsample_target_obj.float())
        #Regressor Loss
        #Multiplying p* with subsample_target_obj to remove 0's bounding boxes
        subsample_regr_out = subsample_target_obj*subsample_regr_out
        subsample_targ_regr = subsample_target_obj*subsample_targ_regr
        # print("subsample_regr_out",subsample_regr_out.shape)
        # print("subsample_targ_regr",subsample_targ_regr.shape)
        # print("M value",effective_batch*target_obj.size(0))
        # print("Correct Pos Index",correct_pos_ind.size()[0])

        regressor_loss = self.loss_reg(subsample_regr_out.view(-1),subsample_targ_regr.view(-1))*(M/correct_pos_ind.size()[0])
        total_loss = self.loss_reg(subsample_regr_out.view(-1),subsample_targ_regr.view(-1)) + classifier_loss
        # total_loss = regressor_loss + classifier_loss 
        return classifier_loss,regressor_loss,total_loss



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
       ####################################
       # TODO postprocess a batch of images
       #####################################
        nms_clas_list = []
        nms_prebox_list = []
        for i in range(len(out_c)):
            nms_clas, nms_prebox = self.postprocessImg(out_c[i],out_r[i])
            nms_clas_list.append(nms_clas)
            nms_prebox_list.append(nms_prebox)
        return nms_clas_list, nms_prebox_list


    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        ######################################
        # TODO postprocess a single image
        #####################################
        # anchor_box_values = torch.zeros_like(self.anchors)
        
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(mat_coord.unsqueeze(0), mat_clas.unsqueeze(0), self.anchors)
        decoded_coord, decoded_corners = output_decoding(flatten_coord,flatten_anchors)
        # for i in range(mat_coord.shape[0]):
        #     for j in range(mat_coord.shape[1]):
        #         anchor_box_values[i][j][0] = self.anchors[i][j][0] + self.anchors[i][j][2]*mat_coord[0][i][j]
        #         # anchor_box_values[i][j][1] = self.anchors[i][j][1]*(1+mat_coord[1][i][j])
        #         anchor_box_values[i][j][1] = self.anchors[i][j][1] + self.anchors[i][j][3]*mat_coord[1][i][j]
        #         anchor_box_values[i][j][2] = self.anchors[i][j][2]*torch.exp(mat_coord[2][i][j])
        #         anchor_box_values[i][j][3] = self.anchors[i][j][3]*torch.exp(mat_coord[3][i][j])
        # # anchor_box_decoded = self.decode(mat_coord)
        #         corners = torch.zeros((4,2))
        #         corners[0] = torch.tensor([anchor_box_values[i][j][0] - 0.5*anchor_box_values[i][j][2], anchor_box_values[i][j][1] + 0.5*anchor_box_values[i][j][3]]) # top_left
        #         corners[1] = torch.tensor([anchor_box_values[i][j][0] + 0.5*anchor_box_values[i][j][2], anchor_box_values[i][j][1] + 0.5*anchor_box_values[i][j][3]]) # to right
        #         corners[2] = torch.tensor([anchor_box_values[i][j][0] + 0.5*anchor_box_values[i][j][2], anchor_box_values[i][j][1] - 0.5*anchor_box_values[i][j][3]]) # bottom right
        #         corners[3] = torch.tensor([anchor_box_values[i][j][0] - 0.5*anchor_box_values[i][j][2], anchor_box_values[i][j][1] - 0.5*anchor_box_values[i][j][3]])
        #         # anchor_box_values[i][j] = torch.logical_and(torch.nonzero((corners[:,0]>0 and corners[:,0]<1088) and (corners[:,1]>0 and corners[:,1]<800))) # anchor boes which are inside image
                # if len((corners[:,0]<0).nonzero())!=0 or len((corners[:,0]>1088).nonzero())!=0 or len((corners[:,1]<0).nonzero())!=0 or len((corners[:,1]>800).nonzero())!=0:
                #     mat_clas[0][i][j] = 0

        indices = (decoded_corners[:,0]<0).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,0]>1088).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,2]>1088).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,2]<0).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,1]<0).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,1]>800).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,3]>800).nonzero()
        flatten_gt[indices] = 0
        indices = (decoded_corners[:,3]<0).nonzero()
        flatten_gt[indices] = 0
        # if len((corners[:,0]<0).nonzero())!=0 or len((corners[:,0]>1088).nonzero())!=0 or len((corners[:,1]<0).nonzero())!=0 or len((corners[:,1]>800).nonzero())!=0:
        #             mat_clas[0][i][j] = 0

        m = flatten_gt.view(-1).argsort(0,descending=True)
        nms_clas, nms_prebox = self.NMS(flatten_gt[m[0:keep_num_preNMS]],
                                        decoded_coord[m[0:keep_num_preNMS],:],
                                        IOU_thresh,
                                        keep_num_postNMS)
        # keep the top K objectiveness scores
        
        return nms_clas, nms_prebox


    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh, keep_num_postNMS):
        ##################################
        # TODO perform NMS
        ##################################
        x_c = prebox[:,0]
        y_c = prebox[:,1]
        w = prebox[:,2]
        h = prebox[:,3]
        bboxes = [x_c-w/2, y_c-h/2, x_c+w/2, y_c+h/2]
        
        iou_matrix = IOU_matrix(bboxes)
        # pdb.set_trace()
        bbox_group = (iou_matrix>thresh).nonzero()

        # remove duplicate entires of the iou matrix (which are mirrored abot the right diagonal)
        for g in range(bbox_group.shape[0]):
          h = g+1
          while h<len(bbox_group):
            if(bbox_group[g][0] == bbox_group[h][1] and bbox_group[g][1] == bbox_group[h][0] and g!=h):
              bbox_group = torch.cat([bbox_group[0:h], bbox_group[h+1:]])
              h = h-1
            h = h+1
        # pdb.set_trace()
        nms_clas = clas
        if bbox_group.shape[0]:
          ptr = 0
          while(ptr<clas.shape[0]):
            p_ = (bbox_group[:,0] == ptr).nonzero().T
            group = bbox_group[p_,1].view(-1)
            conf_max = group[0]
            for k in group:
              if nms_clas[k] > nms_clas[conf_max]:
                    conf_max = k
            #   if nw_output[i_0[k], 0, i_1[k], i_2[k]] > nw_output[i_0[conf_max], 0, i_1[conf_max], i_2[conf_max]]:
            #     conf_max = k
            ptr = ptr+1
            for j in group:
              if j!=conf_max:
              #   nw_output[i_0[j], 0, i_1[j], i_2[j]] = 0
                  nms_clas[j] = 0


        n = nms_clas.view(-1).argsort(0,descending=True)
        # nms_indices = torch.cat(((n // 68).view(-1, 1), (n % 68).view(-1, 1)), dim=1)
        # pdb.set_trace()
        # nms_clas_post = nms_clas[0,nms_indices[0:keep_num_postNMS,0],nms_indices[0:keep_num_postNMS,1]]
        nms_clas_post = nms_clas[n[0:keep_num_postNMS]]
        nms_prebox = prebox
        # nms_prebox = nms_prebox[nms_indices[0:keep_num_postNMS,0],nms_indices[0:keep_num_postNMS,1],:]
        nms_prebox = nms_prebox[n[0:keep_num_postNMS],:]
        return nms_clas_post, nms_prebox

    # def NMS2(self):
    #   for i in range()
    
if __name__=="__main__":
    pass