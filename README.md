# 3D scene reconstruction from single image using Object-level Segmentation-Reconstruction-Localization

##  Setup :-

Setup conda environment using our `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate l3d
```

If you do not have Anaconda, you can quickly download it [here](https://docs.conda.io/en/latest/miniconda.html), or via the command line in with:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Download pre-compiled blender from : https://www.blender.org/download/release/Blender3.5/blender-3.5.0-linux-x64.tar.xz/

## Data :-

* Generated using Blender (Pre-compiled version included under blender)
* Annotated bboxes using https://www.makesense.ai/
* Raw scene images under data/raw_images
* Camera/object poses of $i^\text{th}$ scene under data/poses/i.txt with format :-
    * x y z r p y : Camera pose (in m, degrees)
    * n : no of objects
    * n lines 'x y z r p y' : Object pose (in m, degrees)
* GT object bounding boxes of $i^\text{th}$ scene under data/bboxes/i.txt with format :-
    * n : no of objects
    * n lines 'x y w h' : Object bbox with x,y : center coordinate and w,h : width,height (in pixels)
 
## Pipeline :-

1. Run Object detection -> segmentation to get object : Mansi

2. Run DL based camera pose estimation object wise : Pranay/Mansi

3. Run DL based single image to Nerf estimation object wise : Pranay/Praveen

4. Run iNerf (Nerf localiztion to scene given image supervision) : Dvij

## Results :-
