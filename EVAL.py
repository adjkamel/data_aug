import numpy as np
from PIL import Image
import os
from cv2 import cv2
import math

# Euclidean distance and intersection over union evaluation on the testing set of the SPARCS dataset with 8 images
# ./Pred_before_augmentation/ : the path must contain the prediction results of the testing set using the trained model on images before augemntation (original dataset)
# ./Pred_after_augmentation/ : the path must contain the prediction results of the testing set using the trained model on images after augemntation (augmented dataset)
# ./ground_truth/ : the path must contain the 8 clored ground truth images 
# ./ground_truth_grey/ : the path must contain the 8 grey ground truth images with pixels values between 0 and 255

# To generate the prediction results on the testing set, use CGAN.py with the pretrained models (best_weights_before_augmentation.h5  and generator_before_augmentation.h5) or (best_weights_after_augmentation.h5  and generator_after_augmentation.h5)

dir_before = os.listdir("Pred_before_augmentation")
dir_after = os.listdir("Pred_after_augmentation")
dir_ground_truth = os.listdir("ground_truth")
dir_ground_truth_grey = os.listdir("ground_truth_grey")



def IoU(Yi, y_predi):
    
    th = 15  # threshold because the ground images don't have the same pixels values for the same class
    IoUs = []
    classes = [128, 217, 255, 0, 95, 122]

    # grey pixels values: 128 land, 217 ice snow, 255 cloud, 0 cloud shadow, 95 water, 122 Flooded

    for c in classes:
        TP = np.sum(((Yi >= int(c)-th) & (Yi <= int(c)+th)) &
                    ((y_predi < int(c)-th) | (y_predi > int(c)+th)))
        FP = np.sum(((Yi < int(c)-th) | (Yi > int(c)+th)) &
                    ((y_predi >= int(c)-th) & (y_predi <= int(c)+th)))
        FN = np.sum(((Yi >= int(c)-th) & (Yi <= int(c)+th)) &
                    ((y_predi < int(c)-th) | (y_predi > int(c)+th)))

        # In case the ground images have the same pixels values for the same class
        #TP = np.sum((Yi == int(c)) & (y_predi == int(c)))
        #FP = np.sum((Yi != int(c)) & (y_predi == int(c)))
        #FN = np.sum((Yi == int(c)) & (y_predi != int(c)))

        IoU = TP/float(TP + FP + FN)
        if (math.isnan(IoU)):
            IoU = 0
     
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))





acc = 0

for k in range(0, len(dir_before)):




    img_before = cv2.imread(
        'Pred_before_augmentation/'+dir_before[k])

    img_after = cv2.imread(
        'Pred_after_augmentation/'+dir_after[k])

    img_ground_color = cv2.imread(
        'ground_truth/'+dir_ground_truth[k])

    img_ground_grey = cv2.imread(
        'ground_truth_grey/'+dir_ground_truth_grey[k])

   
     
    print('--------------------------------------------------------------- Image:', k+1)

    print('------ Euclidean distance between pixels''s colors,  applied on grount truth grey images of the testing set of 8 images :', k+1)


    dist_before = np.sum(
        np.sqrt(np.sum((np.square(np.asarray(img_before[:, :, 0]) - np.asarray(img_ground_color[:, :, 0])) + (np.asarray(img_before[:, :, 1]) - np.asarray(img_ground_color[:, :, 1])) + (np.asarray(img_before[:, :, 2]) - np.asarray(img_ground_color[:, :, 2]))))))


    dist_after = np.sum(
        np.sqrt(np.sum(np.square((np.asarray(img_after[:, :, 0]) - np.asarray(img_ground_color[:, :, 0])) + (np.asarray(
            img_after[:, :, 1]) - np.asarray(img_ground_color[:, :, 1]))+(np.asarray(img_after[:, :, 2]) - np.asarray(img_ground_color[:, :, 2]))))))

    

    if (dist_after <= dist_before):
        acc = acc+1

    img_before_grey = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_after_grey = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    img_groundgrey = cv2.cvtColor(img_ground_color, cv2.COLOR_BGR2GRAY)

    print("------ intersection over union (IOU) for each image based on grey threshold th, applied on grount truth grey images of the testing set of 8 images:")

    IoU(img_before_grey, img_groundgrey)
    IoU(img_after_grey, img_groundgrey)

    # print(img_b1)


print('-- Euclidean Mean --:', (acc*100)/8)
