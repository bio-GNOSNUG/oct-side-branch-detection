import torch
from torch.utils.data import Dataset
import numpy as np
import json
import cv2


class SB_Dataset_Inc_Negatives(Dataset):

    def __init__(self, data_root, modality, subset, transforms, resolution, t_frames):
        self.transforms = transforms
        self.subset = subset
        self.data_root = data_root
        self.resolution = resolution
        self.modality = modality
        self.t_frames = t_frames

        # load all annotations into memory
        with open(data_root + 'external/annotation_jsons/sb_{}.json'.format(subset), 'r') as j:
            self.annotations = json.loads(j.read())

        # we need to be able to index the annotations in __getitem__ so convert to list
        self.annotations = list(self.annotations.items())

        print('{} set: {} examples'.format(self.subset, len(self.annotations)))

        # classes: 0 index is reserved for background
        self.classes = ['none', 'sidebranch']

    def load_window(self, frame_id, vessel_name):

        if self.t_frames == 1:
            frame_ids = [frame_id]

        else:
            half = self.t_frames // 2
            frame_ids = list(range(frame_id-half, frame_id+half+1))

        images = []
        
        for f in frame_ids:
            
            image_path = self.data_root + 'processed/{}/{}/{}_frames/{}.npy'.format(self.subset, vessel_name,self.modality, f)
            img = np.load(image_path)
            images.append(img) # 5 x (1024,1024)

        return images
        

    def __getitem__(self, idx):

        img_path, anno = self.annotations[idx]

        vessel_name = img_path[:-5]
        frame_id = int(img_path[-4:])

        images = self.load_window(frame_id, vessel_name)

        # Resize images to specified resolution
        images = [
            cv2.resize(img,
                       (self.resolution,self.resolution),
                       interpolation=cv2.INTER_AREA) 
                       for img in images
                       ] # 5 × (224×224)
        
        images = [img / 255.0 for img in images] # [0,1] just like the pretrained weights.
        image = np.stack(images, axis=-1)  # (224,224,5)

        # uses pascal_voc notations (x1,y1,x2,y2)
        if len(anno) > 0:

            boxes = [l[0] for l in anno]
            # convert boxes into a torch.Tensor
            boxes = np.array(boxes, dtype=np.float32)
            # scale bounding boxes to new image size. i.e 1024 to 224.
            if self.modality == 'oct':
                boxes = boxes * (self.resolution / 1024)
            boxes = np.rint(boxes).astype(np.float32)
            # getting the areas of the boxes
            #print(boxes)
            if len(boxes.shape) > 1:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else:
                area = np.zeros((boxes.shape[0],))

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            area = torch.from_numpy(area)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = vessel_name + '_' + frame_id

        else:

            iscrowd = torch.empty((0), dtype=torch.int64)
            labels = torch.empty((0), dtype=torch.int64)
            area = torch.empty((0), dtype=torch.int64)

            target = {}
            target["boxes"] = np.empty((0,4), dtype=np.float32)
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["image_id"] = vessel_name + '_' + frame_id

        #print(target)


        if self.transforms:

            sample = self.transforms(image=image.copy(),
                                     bboxes=target['boxes'],
                                     labels=labels)
            image = sample['image']
            boxes = sample['bboxes']

        image = torch.from_numpy(image).to(torch.float32)
        image = torch.permute(image, (2,0,1)) # (5,224,224)

        if len(anno) > 0:
            target['boxes'] = torch.as_tensor(boxes,dtype=torch.float32)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0), dtype=torch.int64)
        if self.subset != 'test':
            if len(target['boxes'].shape) < 2:
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                print('ERROR: bbox of shape {} in exampe {}'.format(target['boxes'].shape, target["image_id"]))

        #print(target['labels'], target['labels'].dtype)
        #print(image.shape, image.dtype, image.min(), image.max())
        #print(target['boxes'].shape, target['boxes'].dtype, target['boxes'].min(), target['boxes'].max())

        return image, target

    def __len__(self):
        return len(self.annotations)
