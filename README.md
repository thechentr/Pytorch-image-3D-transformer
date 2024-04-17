# Pytorch-image-3D-transformer

This repository utilizes the torchvision.transforms.functional.perspective function to transform adversarial patches within a 3D space, which maintain differentiability with respect to the original image, making it particularly useful for creating patches that are effective in deceiving 3D vision models.

## Patch
![](runs/transformed_image_1.png)  ![](runs/transformed_image_2.png)
![](runs/transformed_image_3.png) ![](runs/transformed_image_4.png)

## Mask
![](runs/transformed_mask_1.png)  ![](runs/transformed_mask_1.png)
![](runs/transformed_mask_3.png) ![](runs/transformed_mask_4.png)
