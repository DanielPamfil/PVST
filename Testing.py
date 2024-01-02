import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from Dataset import get_loader
import Transforms as trans
from torchvision import transforms
import time
from Models.PVST import PVST
from torch.utils import data
import numpy as np
import os

def test_net(args):

    cudnn.benchmark = True
    # Initializes the PVST model
    net = PVST(args)
    net.cuda()
    net.eval()

    # Load the model
    model_path = args.save_model_dir + 'PVST.pth'
    if not os.path.exists(model_path):
        model_path = args.save_model_dir + 'PVST_Final.pth'
    print(model_path)
    # Loads the model state dictionary
    state_dict = torch.load(model_path)

    # Adjusts the state dictionary for models trained on multiple GPUs
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # Loads the adjusted state dictionary
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))

    # Processes each specified test dataset
    test_paths = args.test_paths.split('+')
    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))
        # List to store time taken for each image
        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, image_w, image_h, image_path = data_batch
            images = Variable(images.cuda())

            starts = time.time()
            # Forward pass
            outputs_saliency = net(images)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)
            # Unpacks the output saliency maps
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_s = F.sigmoid(mask_1_1)

            output_s = output_s.data.cpu().squeeze(0)

            # Transforms the output to match the input image size
            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = transform(output_s)
            # Saves the saliency map to the specified directory
            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/PVST/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            output_s.save(os.path.join(save_test_path, filename + '.png'))

        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list) * 1000))



