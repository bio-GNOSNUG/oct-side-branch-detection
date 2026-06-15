from src.datasets.sb_dataset import SB_Dataset
from src.datasets.sb_dataset_inc_negatives import SB_Dataset_Inc_Negatives
from torch.utils.data import DataLoader
from src.datasets.augmentations import get_transform
from detection_utils.utils import collate_fn, collate_fn_inc_negatives

def load_dataset(config):

    data_root = config['DATA_ROOT']

    if config['DATASET'] == 'sb':
        train_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                   modality=config['MODALITY'],
                                   subset='train',
                                   transforms=get_transform(train=config['AUGMENTATION']),
                                   resolution=config['RESOLUTION'])
        val_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                 modality=config['MODALITY'],
                                 subset='val',
                                 transforms=False,
                                 resolution=config['RESOLUTION'])
        test_dataset = SB_Dataset(data_root=data_root + 'Side Branch Annotations/',
                                  modality=config['MODALITY'],
                                  subset='test',
                                  transforms=False,
                                  resolution=config['RESOLUTION'])

        train_loader = DataLoader(train_dataset,
                                  batch_size=config['TRAIN_BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=config['NUM_WORKERS'],
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True,
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['VAL_BATCH_SIZE'],
                                 num_workers=config['NUM_WORKERS'],
                                 pin_memory=True,
                                 collate_fn=collate_fn)

        return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)

    elif config['DATASET'] == 'sb_inc_negatives':
        train_dataset = SB_Dataset_Inc_Negatives(data_root=data_root + 'Side Branch Annotations/',
                                   modality=config['MODALITY'],
                                   subset='train',
                                   transforms=get_transform(train=config['AUGMENTATION']),
                                   resolution=config['RESOLUTION'])
        val_dataset = SB_Dataset_Inc_Negatives(data_root=data_root + 'Side Branch Annotations/',
                                 modality=config['MODALITY'],
                                 subset='val',
                                 transforms=False,
                                 resolution=config['RESOLUTION'])
        test_dataset = SB_Dataset_Inc_Negatives(data_root=data_root + 'Side Branch Annotations/',
                                  modality=config['MODALITY'],
                                  subset='test',
                                  transforms=False,
                                  resolution=config['RESOLUTION'])

        train_loader = DataLoader(train_dataset,
                                  batch_size=config['TRAIN_BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=config['NUM_WORKERS'],
                                  collate_fn=collate_fn_inc_negatives)
        val_loader = DataLoader(val_dataset,
                                batch_size=config['VAL_BATCH_SIZE'],
                                num_workers=config['NUM_WORKERS'],
                                pin_memory=True,
                                collate_fn=collate_fn_inc_negatives)
        test_loader = DataLoader(test_dataset,
                                 batch_size=config['VAL_BATCH_SIZE'],
                                 num_workers=config['NUM_WORKERS'],
                                 pin_memory=True,
                                 collate_fn=collate_fn_inc_negatives)

        return (train_loader, val_loader, test_loader), (train_dataset, val_dataset, test_dataset)


    else:
        print('WARNING - No dataset selected..')

