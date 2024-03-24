# Person-in-Place Demo & Tutorial Folder
Demo and Tutorial are provided in three ways.  
* The first is through a demo code provided in a ".py" file.  
* The second is through an "ipynb" (Jupyter Notebook) file.  
* The third is by using the "Hugging Face" platform. 

Currently, .py and .ipynb have been implemented, but the hugging face is being built quickly and has not been updated yet.

## Running .ipynb for Tutorial
### Our model's behavior on high-resolution image
### Make Object interactive Skeleton Tutorial
![front_figur](../assets/output2.gif)


We provide tutorial code for extracting multi-person skeletons from real high-resolution  
You can see it by checking the file [here](high-resolution_tutorial.ipynb).  
Additionally, we have placed the pre-defined skeleton in the "pre_defined_skeleton".  
#### Feel free to customize and use it according to your preferences at any time.

### Make HOI image editing Tutorial

![front_figur](../assets/output3.gif)


We conduct high-resolution Human-Object Interaction (HOI) image editing using real images, based on the “pre-defined_skeleton”  
You can see it by checking the file [here](high-resolution_inpainting_tutorial.ipynb).  
Additionally, we provide the prompt we used in the main paper.
#### Feel free to customize and use it according to your preferences at any time.


### Make Object interactive Skeleton Tutorial

![front_figur](../assets/output.gif)

You can see it by checking the file [here](Person-in-Place_demo.ipynb).  
This file includes preliminary code for extracting the skeleton from the background image, along with visualization results.   
Additionally, you can explore various visualization outcomes by examining the saved results in the 'various_results' folder.  

### Make HOI image editing Tutorial

![front_figur](../assets/output1.gif)

You can see it by checking the file [here](Person-in-Place_demo_1.ipynb).  

This tutorial elucidates ControlNet inpainting based on our skeleton results.  
It is recommended to proceed directly after completing the experiment outlined in the previous 'Make Object Interactive Skeleton Tutorial.'  

Additionally, it's important to note that the results may slightly deviate from '.py' Demo.  
This discrepancy arises from the nature of ControlNet inpainting, which necessitates setting the resolution to 512.   
The distinction lies in whether the image size is initially set to the image size or if the skeleton is produced with the resolution set to 512.   
A comprehensive explanation is available in the tutorial.  

## Running SD-Inpainting & SDXL-Inpainting for demo
SD-inpainting and SDXL-inpainting compared to our model will be updated!

# Running .py for demo
### Make Object interactive Skeleton Demo

```
python Person-in-Place_Demo.py --gpu 0 --model_path ../output/SOTA/checkpoint/snapshot_30.pth.tar --cfg ../assets/yaml/v-coco_diffusion_image_feature_demo.yml
```
These results are saved in the 'various_results_512' folder.

### Make HOI image editing
and your conda environment setting is "HOI_editing_cont"

```
python Person-in-Place_Demo_1.py
```
These results are saved in the 'inp_image_512' folder.

## high-resolution image
### Make Object interactive Skeleton Demo
```
python high-resolution_Demo.py --gpu 0 --model_path ../output/SOTA/checkpoint/snapshot_30.pth.tar --cfg ../assets/yaml/v-coco_diffusion_image_feature_demo.yml --json_name ./pre_define_skeleton/bedroom_demo_1.json --image_name bedroom_demo_1.jpg
```

### Make HOI image editing
and your conda environment setting is "HOI_editing_cont"
```
python high-resolution_inpainting_Demo.py
```

# Running Hugging Face
Hugging Face demo will be updated!