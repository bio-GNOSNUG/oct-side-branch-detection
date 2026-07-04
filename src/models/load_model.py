import os
import torch
from src.models.faster_rcnn_edit import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn, adapt_input_conv_weights

def load_model(config, device):

    pretrained = config["PRETRAINED_WEIGHTS"] is not None

    if config['MODEL'] == 'fasterrcnn_resnet18_fpn':

        if config['INFERENCE']:
            model = fasterrcnn_resnet18_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            pretrained=pretrained,
                                            resolution=config["RESOLUTION"],
                                            input_dim=config["INPUT_DIM"],
                                            # rpn_fg_iou_thresh=config['RPN_FG_IOU_THRESH'], # train
                                            # rpn_bg_iou_thresh=config['RPN_BG_IOU_THRESH'], # train
                                            rpn_pre_nms_top_n_test=config['PRE_NMS_TOP_N_TEST'], # test
                                            box_nms_thresh=config['NMS_IOU'],
                                            box_score_thresh=config['CONF_THRESH']
                                            ).to(device)

        else:
            model = fasterrcnn_resnet18_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            pretrained=pretrained,
                                            resolution=config["RESOLUTION"],
                                            input_dim=config["INPUT_DIM"]
                                            ).to(device)

    elif config['MODEL'] == 'fasterrcnn_resnet50_fpn':

        if config['INFERENCE']:
            model = fasterrcnn_resnet50_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            pretrained=pretrained,
                                            resolution=config["RESOLUTION"],
                                            input_dim=config["INPUT_DIM"],
                                            # rpn_fg_iou_thresh=config['RPN_FG_IOU_THRESH'], # train
                                            # rpn_bg_iou_thresh=config['RPN_BG_IOU_THRESH'], # train
                                            rpn_pre_nms_top_n_test=config['PRE_NMS_TOP_N_TEST'], # test
                                            box_nms_thresh=config['NMS_IOU'],
                                            box_score_thresh=config['CONF_THRESH']
                                            ).to(device)

        else:
            model = fasterrcnn_resnet50_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            pretrained=pretrained,
                                            resolution=config["RESOLUTION"],
                                            input_dim=config["INPUT_DIM"]
                                            ).to(device)

    else:
        print('Model not recognised')


    if pretrained:

        weight_path = os.path.join('tests/results', config["PRETRAINED_WEIGHTS"], 'best_map.pt')
        state_dict = torch.load(weight_path,map_location=device)
        state_dict = adapt_input_conv_weights(state_dict,config["INPUT_DIM"])

        model.load_state_dict(state_dict, strict=False)


    return model