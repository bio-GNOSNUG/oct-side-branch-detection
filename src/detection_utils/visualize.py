import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def plot_loss(train_loss, val_loss, val_map, folder):
    epochs = len(val_loss)
    f, axes = plt.subplots(1,2, figsize=(20,10))
    axes[0].plot(list(range(epochs)), train_loss, label='train_loss')
    axes[0].plot(list(range(epochs)), val_loss, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(0, 0.4)
    axes[1].plot(list(range(epochs)), val_map, label='val mAP')
    axes[1].set_title('mAP')
    axes[1].legend()
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('mAP')
    axes[1].set_ylim(0, 0.8)
    plt.savefig(folder + '/progress.png', dpi=200)
    plt.close('all')


def plot_img_bbox(img, target, pred, save_path, low_resolution):

    if target is not None:
        target_box = target['boxes']
        if torch.is_tensor(target_box):
            if target_box.is_cuda: target_box = target_box.cpu()

    pred_box = pred['boxes']
    if torch.is_tensor(pred_box):
        if pred_box.is_cuda: pred_box = pred_box.cpu()

    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    if low_resolution:
        fig, a = plt.subplots(1,1, figsize=(4,4))
    else:
        fig, a = plt.subplots(1, 1, figsize=(8,8))
    a.imshow(img)
    if target is not None:
        for box in target_box:
            x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth = 1,
                                     edgecolor = 'g',
                                     facecolor = 'none')
            a.add_patch(rect)

    for box in pred_box:
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        a.add_patch(rect)

    plt.axis('off')
    if low_resolution:
        plt.savefig(save_path, dpi=50, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close('all')