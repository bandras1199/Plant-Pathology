# Plant-Pathology
Experimenting with the Kaggle Plant Pathology Dataset: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

Initial approach: <br />

  I had very limited resources for this project (google colab + my old laptop) therefore I wanted 
  to reduce the computational needs as much as possible, using lots of image processing and trying to 
  find a good performing but still sustainable network architecture.
  
<strong> Image processing steps: </strong> 

  1, Initial image:
  
 <img src="/demo/0orig.png" alt="description" height="175" width="300" />   
 
  2, Removing blurry background noise:
  - Left image: Applying Fourier transormation on the image and plotting the magnitude spectrum <br />
  - Middle image: Covering the middle (low frequencies) of the magnitude spectrum, resulting a high pass filter <br />
  - Right image: Removing low frequency parts (blurry background), plotted in JET [1] <br />
 
 <img src="/demo/1mag_orig.png" alt="description" height="175" width="300" /> <img src="/demo/2mag_cube.png" alt="description" height="175" width="300" /> <img src="/demo/3rem.png" alt="description" height="175" width="300" />
 
 3, Object detection:
 - Left image: Sharping the edges then detecting the outlier contours <br />
 - Middle image: First fit an ellipse on the contours of the leaf, make it parallel to X axis and then fit the biggest possible rectangle inside this ellipse by solving the optimization problem [2] <br />
 - Right image: Final image, cropped to standard size <br />
 
  <img src="/demo/4cont.png" alt="description" height="175" width="300" /> <img src="/demo/5fitellipse4.png" alt="description" height="175" width="300" /> <img src="/demo/6compr.png" alt="description" height="175" width="300" />
  
4, Augmentation: 
- Applying random distortion and rotation on the image, automatically crop the black sides. [3] [4] <br />
- Image: Previous image after augmentation. <br />
 <img src="/demo/7augmentation.png" alt="description" height="175" width="300" />
 
Mistakes: 
- Some images have their infected part close to their edge, which has to be considered during object detection. Currently, these faulty images are excluded from the experiment.
 - Left image: Original image <br />
 - Right image: Processed Image <br />
  <img src="/demo/8mistake2.png" alt="description" height="175" width="300" /> <img src="/demo/8mistake.png" alt="description" height="175" width="300" /> 

<strong> Deep Learning: </strong> 

 
 
 
 <strong> References: </strong> 
 
 [1] OpenCV Documentation, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
 [2] Calculus Optimization #9, https://www.youtube.com/watch?v=r0wdreyN4QE
 [3] Image Augmentation in Numpy, https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd
 [4] Automatic crop after rotation, https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py 
 
 
 
