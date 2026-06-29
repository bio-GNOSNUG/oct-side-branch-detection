import torchvision
from src.models.faster_rcnn_edit import fasterrcnn_resnet18_fpn, fasterrcnn_resnet50_fpn

def load_model(config, device):

    if config['MODEL'] == 'fasterrcnn_resnet18_fpn':

        if config['INFERENCE']:
            model = fasterrcnn_resnet18_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            resolution=config["RESOLUTION"],
                                            # rpn_fg_iou_thresh=config['RPN_FG_IOU_THRESH'], # train
                                            # rpn_bg_iou_thresh=config['RPN_BG_IOU_THRESH'], # train
                                            rpn_pre_nms_top_n_test=config['PRE_NMS_TOP_N_TEST'], # test
                                            box_nms_thresh=config['NMS_IOU'],
                                            box_score_thresh=config['CONF_THRESH']
                                            ).to(device)

        else:
            model = fasterrcnn_resnet18_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            resolution=config["RESOLUTION"]
                                            ).to(device)

    elif config['MODEL'] == 'fasterrcnn_resnet50_fpn':

        if config['INFERENCE']:
            model = fasterrcnn_resnet50_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            resolution=config["RESOLUTION"],
                                            # rpn_fg_iou_thresh=config['RPN_FG_IOU_THRESH'], # train
                                            # rpn_bg_iou_thresh=config['RPN_BG_IOU_THRESH'], # train
                                            rpn_pre_nms_top_n_test=config['PRE_NMS_TOP_N_TEST'], # test
                                            box_nms_thresh=config['NMS_IOU'],
                                            box_score_thresh=config['CONF_THRESH']
                                            ).to(device)

        else:
            model = fasterrcnn_resnet50_fpn(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            resolution=config["RESOLUTION"]
                                            ).to(device)

    else:
        print('Model not recognised')

    return model