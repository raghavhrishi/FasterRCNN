import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
# from utils import *
import matplotlib.pyplot as plt
# from rpn import *
import matplotlib.patches as patches
import matplotlib.patches as patches
import pdb
import torchvision 
from matplotlib.patches import Rectangle

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
# from utils import *
import matplotlib.pyplot as plt
# from rpn import *
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
      self.images_h5 = h5py.File(path[0],mode='r')
      self.image_data = self.images_h5.get('data')
      self.mask_h5 = h5py.File(path[1],mode='r')
      self.mask_data = self.mask_h5.get('data')
      self.bbox_data = np.load(path[3], allow_pickle=True)
      self.label_data = np.load(path[2], allow_pickle=True)
      self.transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      self.totensor = transforms.ToTensor()
      self.pad = 11
      self.target_h = 800
      self.target_w = 1088
      self.resize_h = self.target_h
      self.resize_w = self.target_w - 2*self.pad
      self.struct_mask = self.grouping_mask_label(self.label_data, self.mask_data)

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox

    def grouping_mask_label(self,labels, masks):
        mask_group =[]
        items = 0
        for i in range(len(labels)):        
            mask_group.append(masks[items:items+len(labels[i])])

            items += len(labels[i])
        return np.array(mask_group)


    def __getitem__(self, index):
        # TODO: __getitem__
        # check flag
        image_data = self.image_data[index]
        label_data = self.label_data[index]
        bbox_data = self.bbox_data[index]
        mask_data = self.mask_data[index]
        mask_list = self.struct_mask[index]
        image_data,mask_list ,bbox_data = self.pre_process_batch(image_data,mask_list,bbox_data)

        assert image_data.shape == (3, 800, 1088)

        assert bbox_data.shape[0] == mask_list.shape[0]


        return image_data,label_data,mask_list,bbox_data,index

    def __len__(self):
        return len(self.image_data)
    
    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        bbox1 =np.zeros(bbox.shape)
        ori_h, ori_w = 300, 400
        transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = torch.tensor(img.astype(float)).unsqueeze(0)        
        img = transform(F.interpolate(img.clone(), size=(800, 1066), mode='nearest').squeeze(0))
        img = F.pad(img, (11,11), 'constant', 0) # padding on both the sides
        mask = torch.tensor(mask.astype(float)).unsqueeze(0)
        mask = F.interpolate(mask, size=(800, 1066), mode='nearest').squeeze(0)
        mask = F.pad(mask, (11,11), 'constant', 0)
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        x_scale = 800 / 300
        y_scale = 1066 / 400
        bbox1[:, 0] = bbox[:, 0] * 2.665  + self.pad
        bbox1[:, 1] = bbox[:, 1] * 2.6667
        bbox1[:, 2] = bbox[:, 2] * 2.665 + self.pad
        bbox1[:, 3] = bbox[:, 3] * 2.6667

        return img, mask, bbox1

class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers      

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        # TODO: collect_fn
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        index_list = []
        for transed_img, label, transed_mask, transed_bbox,index in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
            index_list.append(index)
        return {"images":torch.stack(transed_img_list, dim=0), "label" :label_list, "mask":transed_mask_list, "bbox":transed_bbox_list,"index":index_list}
      

    def loader(self):
      return DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True, collate_fn=self.collect_fn)


# class BuildDataset(torch.utils.data.Dataset):
#     def __init__(self, path):
#       self.images_h5 = h5py.File(path[0],mode='r')
#       self.image_data = self.images_h5.get('data')
#       self.mask_h5 = h5py.File(path[1],mode='r')
#       self.mask_data = self.mask_h5.get('data')
#       self.bbox_data = np.load(path[3], allow_pickle=True)
#       self.label_data = np.load(path[2], allow_pickle=True)
#       self.transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#       self.totensor = transforms.ToTensor()
#       self.pad = 11
#       self.target_h = 800
#       self.target_w = 1088
#       self.resize_h = self.target_h
#       self.resize_w = self.target_w - 2*self.pad

#       self.struct_mask = self.grouping_mask_label(self.label_data, self.mask_data)

#     # output:
#         # transed_img
#         # label
#         # transed_mask
#         # transed_bbox

#     def grouping_mask_label(self,labels, masks):
#         mask_group =[]
#         items = 0
#         for i in range(len(labels)):        
#             mask_group.append(masks[items:items+len(labels[i])])

#             items += len(labels[i])
#         return np.array(mask_group)


#     def __getitem__(self, index):
#         # TODO: __getitem__
#         # check flag
#         image_data = self.image_data[index]
#         label_data = self.label_data[index]
#         bbox_data = self.bbox_data[index]
#         mask_data = self.mask_data[index]
#         mask_list = self.struct_mask[index]
#         image_data,mask_list ,bbox_data = self.pre_process_batch(image_data,mask_list,bbox_data)

#         assert image_data.shape == (3, 800, 1088)

#         assert bbox_data.shape[0] == mask_list.shape[0]


#         return image_data,label_data,mask_list,bbox_data

#     def __len__(self):
#         return len(self.image_data)
    
#     # This function take care of the pre-process of img,mask,bbox
#     # in the input mini-batch
#     # input:
#         # img: 3*300*400
#         # mask: 3*300*400
#         # bbox: n_box*4
#     def pre_process_batch(self, img, mask, bbox):
#         bbox1 =np.zeros(bbox.shape)
#         ori_h, ori_w = 300, 400
#         transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#         img = torch.tensor(img.astype(float)).unsqueeze(0)        
#         img = transform(F.interpolate(img.clone(), size=(800, 1066), mode='nearest').squeeze(0))
#         img = F.pad(img, (11,11), 'constant', 0) # padding on both the sides
#         mask = torch.tensor(mask.astype(float)).unsqueeze(0)
#         mask = F.interpolate(mask, size=(800, 1066), mode='nearest').squeeze(0)
#         mask = F.pad(mask, (11,11), 'constant', 0)
#         assert img.shape == (3, 800, 1088)
#         assert bbox.shape[0] == mask.shape[0]
#         x_scale = 800 / 300
#         y_scale = 1066 / 400
#         bbox1[:, 0] = bbox[:, 0] * 2.665  + self.pad
#         bbox1[:, 1] = bbox[:, 1] * 2.6667
#         bbox1[:, 2] = bbox[:, 2] * 2.665 + self.pad
#         bbox1[:, 3] = bbox[:, 3] * 2.6667

#         return img, mask, bbox1

# class BuildDataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size, shuffle, num_workers):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_workers = num_workers      

#     # output:
#         # img: (bz, 3, 800, 1088)
#         # label_list: list, len:bz, each (n_obj,)
#         # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
#         # transed_bbox_list: list, len:bz, each (n_obj, 4)
#         # img: (bz, 3, 300, 400)
#     def collect_fn(self, batch):
#         # TODO: collect_fn
#         transed_img_list = []
#         label_list = []
#         transed_mask_list = []
#         transed_bbox_list = []
#         for transed_img, label, transed_mask, transed_bbox in batch:
#             transed_img_list.append(transed_img)
#             label_list.append(label)
#             transed_mask_list.append(transed_mask)
#             transed_bbox_list.append(transed_bbox)
#         return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list
      

#     def loader(self):
#       return DataLoader(self.dataset, batch_size=self.batch_size,shuffle=True, collate_fn=self.collect_fn)

# if __name__ == '__main__':
#     # file path and make a list
#     imgs_path = '/content/gdrive/My Drive/hw3_mycocodata_img_comp_zlib.h5'
#     masks_path = '/content/gdrive/My Drive/hw3_mycocodata_mask_comp_zlib.h5'
#     labels_path = '/content/gdrive/My Drive/hw3_mycocodata_labels_comp_zlib.npy'
#     bboxes_path = '/content/gdrive/My Drive/hw3_mycocodata_bboxes_comp_zlib.npy'
#     paths = [imgs_path, masks_path, labels_path, bboxes_path]
#     # load the data into data.Dataset
#     dataset = BuildDataset(paths)
#     # build the dataloader
#     # set 20% of the dataset as the training data
#     full_size = len(dataset)
#     train_size = int(full_size * 0.8)
#     test_size = full_size - train_size
#     # random split the dataset into training and testset
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#     # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
#     # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
#     batch_size = 1
#     train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     train_loader = train_build_loader.loader()
#     test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader = test_build_loader.loader()
#     mask_color_list = [ "Spectral", "ocean","jet", "Spectral", "ocean"]
#     bbox_color_list = ['b','g','b','r','b']
#     for iter,batch in enumerate(train_loader,0):
#         #img, label_list, mask_list, bbox_list = [batch[iter] for iter in range(len(batch))]
#         img, label, mask, bbox = [batch[iter] for iter in range(len(batch))]
#         for i in range(batch_size):
#           denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.225])
#           #Image Plotting
#           print("Image Size",img[i].shape )
#           image_new = denormalize(img[i].squeeze(0))
#           image_new = np.copy(image_new.cpu().detach().numpy().transpose(1,2,0))
#           plt.imshow(image_new.astype('long'))
#           print("Mask",mask[i].shape)
#           print("Label shape",label[i].shape)
#           print("Label Val",label[i])
#           print("BBox shape",bbox[i].shape)
#           label_shape = label[i].shape
#           channel_dim = mask[i].shape
          
#           if channel_dim[0] == 2:
#             mask1 = mask[i][0]
#             mask1 = mask1.cpu().numpy()
#             masked1 = np.ma.masked_where(mask1 == 0, mask1)
#             colors = mask_color_list[label[i][0].item()]
#             plt.imshow(masked1, cmap=colors, alpha=0.7) 
#             mask2 = mask[i][1]
#             mask2 = mask2.cpu().numpy()
#             masked2 = np.ma.masked_where(mask2 == 0, mask2)
#             colors = mask_color_list[label[i][1].item()]
#             plt.imshow(masked2, cmap=colors, alpha=0.7) 
#           else:
#             print("type",type(mask[i]))
#             masked = np.ma.masked_where(mask[i] == 0, mask[i])
#             colors = mask_color_list[label[i].item()]
#             plt.imshow(masked.squeeze(0), cmap=colors, alpha=0.7) 
#           ax = plt.gca()
          

#           if len(label[i]) == 1:
#             colors = bbox_color_list[label[i].item()]
#             rect2 = patches.Rectangle((bbox[i][0][0],bbox[i][0][1]),bbox[i][0][2] - bbox[i][0][0],bbox[i][0][3] - bbox[i][0][1],linewidth=1,edgecolor=colors,facecolor='none')
#             ax.add_patch(rect2)
#           elif len(label[i]) == 2:
#             colors_1 = bbox_color_list[label[i][0].item()]
#             rect1 = patches.Rectangle((bbox[i][0][0],bbox[i][0][1]),bbox[i][0][2] - bbox[i][0][0],bbox[i][0][3] - bbox[i][0][1],linewidth=1,edgecolor=colors_1,facecolor='none')
#             ax.add_patch(rect1)
#             colors_2 = bbox_color_list[label[i][1].item()]
#             rect2 = patches.Rectangle((bbox[i][1][0],bbox[i][1][1]),bbox[i][1][2] - bbox[i][1][0],bbox[i][1][3] - bbox[i][1][1],linewidth=1,edgecolor=colors_2,facecolor='none')
#             ax.add_patch(rect2)



            
          
#           plt.show()