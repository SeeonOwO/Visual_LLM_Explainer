import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.modeling.poolers import ROIPooler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Boxes


class RelativePositionEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding = nn.Linear(4, embedding_size)

    def forward(self, relative_positions):
        return self.embedding(relative_positions)


class ImageFeaturizer(object):
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = build_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.eval()
        
        output_size = (7, 7)
        scales = (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = 0
        roi_pooler = ROIPooler(output_size, scales, sampling_ratio, pooler_type="ROIAlign")

        self.cfg = cfg
        self.model = model
        self.roi_pooler = roi_pooler
        self.device = cfg.MODEL.DEVICE
        self.max_objects = 59
        # self.position_embedding = RelativePositionEmbedding(128).to(self.device)

    def get_image_features(self, image):
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        image_list = ImageList.from_tensors([image_tensor], self.model.backbone.size_divisibility).to(self.device)
        with torch.no_grad():
            backbone_features = self.model.backbone(image_list.tensor)
            proposals, _ = self.model.proposal_generator(image_list, backbone_features)
            predictions, _ = self.model.roi_heads(image_list, backbone_features, proposals)
            final_instances = predictions[0]
            feature_maps = [backbone_features[f] for f in self.model.roi_heads.in_features]
            boxes_tensor = final_instances.pred_boxes.tensor
            boxes_list = [Boxes(boxes_tensor)]
            detected_object_features = self.roi_pooler(feature_maps, boxes_list)

        traffic_related_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 6: 'truck', 9: 'traffic light', 12: 'stop sign'
        }

        object_features = []
        bbox = []
        for i, (cls, box) in enumerate(zip(final_instances.pred_classes.cpu().numpy(), boxes_tensor.cpu().numpy())):
            if cls in traffic_related_classes.keys():
                class_idx = list(traffic_related_classes.keys()).index(cls)
                class_label = F.one_hot(torch.tensor([class_idx], device=self.device), 
                                        num_classes=len(traffic_related_classes) + 1).float().squeeze(0)
                box_normalized = torch.from_numpy(
                    box / np.array([image.width, image.height, image.width, image.height])).to(self.device).float()
                bbox.append(box_normalized)
                # relative_embeddings = self.position_embedding(box_normalized)
                object_feature = torch.mean(detected_object_features[i], dim=[1, 2]).flatten()
                # feature = torch.cat((class_label, relative_embeddings, object_feature))
                feature = torch.cat((class_label, object_feature))
                object_features.append(feature)

        global_class_idx = len(traffic_related_classes)
        global_class_label = F.one_hot(torch.tensor([global_class_idx], device=self.device), 
                                       num_classes=len(traffic_related_classes) + 1).float().squeeze(0)
        global_box_normalized = torch.tensor([0, 0, 1, 1], dtype=torch.float32).to(self.device)
        bbox.append(global_box_normalized)
        # global_embeddings = self.position_embedding(global_box_normalized)
        global_feature = torch.mean(backbone_features["p5"], dim=[2, 3]).flatten().to(self.device)
        # global_feature_vector = torch.cat((global_class_label, global_embeddings, global_feature))
        global_feature_vector = torch.cat((global_class_label, global_feature))
        object_features.insert(0, global_feature_vector)

        if len(object_features) < self.max_objects + 1:
            padding = torch.zeros(self.max_objects + 1 - len(object_features), object_features[0].shape[0]).to(self.device)
            object_features = torch.cat((torch.stack(object_features), padding), dim=0)
            padding_box = torch.zeros(self.max_objects + 1 - len(bbox), 4).to(self.device)
            bbox_tensor = torch.cat((torch.stack(bbox), padding_box), dim=0)
        else:
            object_features = torch.stack(object_features[:self.max_objects + 1])
            bbox_tensor = torch.stack(bbox[:self.max_objects + 1])

        return object_features, bbox_tensor
