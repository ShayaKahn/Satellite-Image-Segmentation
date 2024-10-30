# Land Cover classification with U-Net (multi-class semantic segmentation)

This repository contains the code for the Land Cover classification using U-Net architecture. 
I used dataset consists of 803 satellite images and the corresponding masks (lables), as shown in the dataset
description chapter in the notebook. I used two preprocessing techniques:

1. Data Augmentation:
    - Random rotation
    - Horizontal flip
    - Vertical flip
    - Random crop

2. Image split:
    - Split the image into 4 parts and then resize.

The logic behind the image split is to increase the image resolution before resizing it to the desired size. 
This will help to maintain the image quality and reduce the information loss. The disadvantage of this technique
is that it will increase the training time and possible information loss in the boundaries due to the image split.
I compared three optional preprocessing techniques: Data Augmentation, Image split and no preprocessing.
For all the options I resized the images to 256x256 pixels. In addition, I compared different loss
functions:

1. Catagorical Crossentropy
2. Dice Loss
3. Jaccard Loss
4. Weighted Dice Loss
5. Weighted Jaccard Loss
6. Combined Loss (Catagorical Crossentropy + Dice Loss/ Jaccard Loss)

I measured the performance of the model using the mean IoU (Intersection over Union). The following table shows
the results:

### No Preprocessing
| Loss Function                                    | Mean IoU (test set) | # of Epochs | Batch Size |
|--------------------------------------------------|---------------------|-------------|------------|
| Catagorical Crossentropy                         | 0.46                | 100         | 32         |
| Catagorical Crossentropy + Dice Loss             | 0.53                | 100         | 32         |
| Catagorical Crossentropy + Jaccard Loss          | 0.56                | 100         | 32         |
| Catagorical Crossentropy + Weighted Dice Loss    |                     |
| Catagorical Crossentropy + Weighted Jaccard Loss |                     |

### With Data Augmentation
| Loss Function                                    | Mean IoU (test set) | # of Epochs | Batch Size |
|--------------------------------------------------|---------------------|-------------|------------|
| Catagorical Crossentropy                         | 0.5                 | 100         | 8          |
| Catagorical Crossentropy + Dice Loss             |                     |
| Catagorical Crossentropy + Jaccard Loss          |                     |
| Catagorical Crossentropy + Weighted Dice Loss    |                     |
| Catagorical Crossentropy + Weighted Jaccard Loss |                     |

### With Image Split
| Loss Function                                    | Mean IoU (test set) | # of Epochs |
|--------------------------------------------------|--| --- |
| Catagorical Crossentropy                         |  |              |  |
| Catagorical Crossentropy + Dice Loss             |  |
| Catagorical Crossentropy + Jaccard Loss          |  |
| Catagorical Crossentropy + Weighted Dice Loss    |  |
| Catagorical Crossentropy + Weighted Jaccard Loss |  |