import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from charades_fileparser import charadesClassParser

class CharadesDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 video_dir, 
                 classes_file, 
                 object_classes_file, 
                 verb_classes_file, 
                 mapping_file, 
                 num_frames, 
                 image_size, 
                 is_training):
        
        self.annotations = pd.read_csv(annotations_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_training = is_training

        self.class_parser = charadesClassParser(classes_file=classes_file, 
                                                object_classes_file=object_classes_file, 
                                                verb_classes_file=verb_classes_file, 
                                                mapping_file=mapping_file)
        #parse actions from annotations
        self.annotations['parsed_actions'] = self.annotations['actions'].apply(self._parse_actions)

        self.transform = self._build_transforms()
        self.class_parser.print_class_summary()

    def _parse_actions(self, actions_str):
        #parse actions string such as "c092 11.90 21.20;c147 0.00 12.60"
        if pd.isna(actions_str) or actions_str == '':
            return []

        actions = []
        for action in actions_str.split(';'):
            parts = action.strip().split()
            if len(parts) >= 3:
                actions.append({'class_id': parts[0], 'start': float(parts[1]), 'end': float(parts[2])})
        return actions

    def _build_transforms(self):
        #build a data augmentation pipeline
        if self.is_training == True:
            return A.Compose([A.Resize(self.image_size, self.image_size),
                              A.HorizontalFlip(p=0.5),
                              A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2(),
                             ])
        else:
            return A.Compose([A.Resize(self.image_size, self.image_size),
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2(),
                             ])

    def __len__(self):
        return len(self.annotations)

    def _load_video_frames(self, video_path):
        #load frames from video using openCV
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return self._create_black_frames()

            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                while len(frames) < self.num_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frames.append(frame)
            else:
                frame_idxs = np.linspace(0, total_frames -1, self.num_frames, dtype=int)
                for frame_idx in frame_idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.image_size, self.image_size))
                        frames.append(frame)
                    else:
                        frames.append(self._create_black_frame())

            cap.release()

            if len(frames) < self.num_frames:
                frames.extend([self._create_black_frame()] * (self.num_frames - len(frames)))
            return frames[:self.num_frames]
            
        except Exception as e:
            print(f"Error loading video: {video_path}: {e}")
            return self._create_black_frames()

    def _create_black_frame(self):
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _create_black_frames(self):
        return [self._create_black_frame() for _ in range(self.num_frames)]

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_id = row['id']
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        #load frames
        frames = self._load_video_frames(video_path)
        
        #apply transforms
        transformed_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)['image']
            transformed_frames.append(transformed)

        video_tensor = torch.stack(transformed_frames) # channels [T, C, H, W]
        
        #create multi-hot labels for verbs, objects, and actions
        verb_labels = torch.zeros(self.class_parser.get_num_verbs())
        object_labels = torch.zeros(self.class_parser.get_num_objects())
        action_labels = torch.zeros(self.class_parser.get_num_actions())

        if row['parsed_actions']:
            for action in row['parsed_actions']:
                action_id = action['class_id']
                #mark action
                if action_id in self.class_parser.action_to_idx:
                    action_idx = self.class_parser.action_to_idx[action_id]
                    action_labels[action_idx] = 1.0
                    
                verb_tensor, obj_tensor = self.class_parser.action_to_component_tensors(action_id)
                verb_labels += verb_tensor
                object_labels += obj_tensor

        #clip to 1.0 in case multiple actions share same verb/object
        verb_labels = torch.clamp(verb_labels, 0.0, 1.0)
        object_labels = torch.clamp(object_labels, 0.0, 1.0)

        return {'video': video_tensor,
                'verb_labels': verb_labels,
                'object_labels': object_labels,
                'action_labels': action_labels,
                'video_id': video_id
               }

    def get_num_verbs(self):
        return self.class_parser.get_num_verbs()

    def get_num_objects(self):
        return self.class_parser.get_num_objects()

    def get_num_actions(self):
        return self.class_parser.get_num_actions()

    def get_class_parser(self):
        return self.class_parser
            

               