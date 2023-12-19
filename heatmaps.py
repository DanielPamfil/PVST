import cv2
import os
import argparse
from os import listdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--original', default= 'Data/UAV123/bike1/images/', type=str, help='folder with original images')
    parser.add_argument('--mask', default= 'preds/UAV123/PVST/', type=str, help='folder with mask images')
    # get the path/directory
    #folder_dir = "Data/UAV123/bike1/images/"
    #mask_dir = "preds/UAV123/RGB_VST/"

    args = parser.parse_args()

    folder_dir = args.original
    mask_dir = args.mask
    if not os.path.exists(mask_dir + 'heatmaps/'):
        os.makedirs(mask_dir + 'heatmaps/')
    #folder_dir = "Data/DUTS/DUTS-TE/DUTS-TE-Image/"
    #mask_dir = "preds/DUTS/RGB_VST/"
    for images in os.listdir(folder_dir):
        #print(images.split(".")[0])

        # check if the image ends with png
        if (images.endswith(".png") or images.endswith(".jpg")):
            img = cv2.imread(folder_dir + images.split(".")[0] + '.jpg')
            mask = cv2.imread(mask_dir + images.split(".")[0] + '.png')
            heatmap_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
            cv2.imwrite(mask_dir + 'heatmaps/' + images.split(".")[0] + '.png', super_imposed_img)

    """
    img = cv2.imread('Data/UAV123/bike1/images/000001.jpg')
    
    mask = cv2.imread('preds/UAV123/RGB_VST/000001.png')
    
    heatmap_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    
    cv2.imshow('image', super_imposed_img)
    
    cv2.waitKey(0)
    """