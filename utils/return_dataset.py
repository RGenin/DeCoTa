import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args, return_idx=False):
    base_path = './data/txt/%s' % args.dataset
    root = args.root
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    # print("RETURN DATASET IMAGESETFILE :", image_set_file_s)
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    source_dataset = Imagelists_VISDA(image_set_file_s, root=os.path.join(root, args.source),
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=os.path.join(root,args.target),
                                      transform=data_transforms['train'], test=True)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=os.path.join(root,args.target),
                                          transform=data_transforms['val'])

    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=os.path.join(root,args.target),
                                              transform=data_transforms['val'], test=True)

    if return_idx:
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=os.path.join(root,args.target),
                                               transform=data_transforms['test'], test=True)
        shuffle_flag = False
    else:
        target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=os.path.join(root, args.target),
                                               transform=data_transforms['test'])
        shuffle_flag = True

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32 # initiallement 32
    else:
        bs = 24 # initiallement : 24

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs, num_workers=3,
                                    shuffle=shuffle_flag, drop_last=False)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list
