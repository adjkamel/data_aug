## Data Augmentation for Land Cover Classification Using Generative Adversarial Networks
This repository contains code for the paper: [Data Augmentation for Land Cover Classification Using Generative Adversarial Networks](https://github.com/csmember/data_aug)

#### Requirement
- Python 3.7
- Tensorflow 2.0
- CUDA Version: 11.1

#### Run
- Input images and labeled images must be resized to 256x256 using RESIZEto256.py
- Run GAN.py to train the model to generate a new images based on a dataset of images (SPARCS) stored in ./images folder. pretrained models will be stored in models_GAN. The images will be stored in generated_images. During the training process you can pick the generated images of the last epochs to augment the dataset.
- RUN CGAN.py to train the CGAN network to generate labels for the original dataset from ./images_CGAN/train/a. The resulting labels generated during training will be stored in ./predict_CGAN. The pretrained models of CGAN will be stored in the root directory ./
- After augmenting the dataset with the new generated images, add them to ./images_CGAN/train/a and Run CGAN.py to generate labeled images. The resulting labels generated during training will be stored in ./predict_CGAN as well.
- To generate labels for the testing set of 8 images in ./images_CGAN/test/a, run CGAN with the pretrained models in ./ root directory before and after augemntation. Use load model in CGAN.py file instead of training from scratch (check the las lines of the code.)
- Link of the dataset:
[SPARCS dataset](https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs)


#### Results
- A screen shot of all 200 generated images. Some of them are in ./gnenerated_images
![generated](https://user-images.githubusercontent.com/50513215/118514193-a6d39a80-b734-11eb-8894-bfd2e887ce8e.PNG)

- To evaluate the models before and after the daa augmentation, put the prediction labeled images generated using the pretrained model of the CGAN before augemntation in the folder ./Pred_before_augmentation, and put the prediction labeled images generated using the pretrained model of the CGAN sfter augemntation in the folder ./Pred_after_augmentation. After that, Run EVAL.py to generate the IOU and the eucledian distance before and after augmentation.

- Comparison results between the classification accuracy before and after data augmentation:
![image](https://user-images.githubusercontent.com/50513215/120845252-73807080-c568-11eb-8884-648ff34cead3.png)
![image](https://user-images.githubusercontent.com/50513215/120846314-d9b9c300-c569-11eb-80cd-16c475863443.png)




- Some of the basic model structure of the GAN design goes to manicman1999/GAN256 
