# Plant-Pathology
Experimenting with the Kaggle Plant Pathology Dataset: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

Initial approach:

  I had very limited resources for this project (google colab + my old laptop) therefore I wanted 
  to reduce the computational needs as much as possible, using lots of image processing and trying to 
  find a good performing but still sustainable network architecture.
  
Image processing steps:

  1, Initial image:
  
 <img src="/demo/0orig.png" alt="description" height="175" width="300" />   
 
  2, Left image: Applying Fourier transormation on the image and plotting the magnitude spectrum
    Middle image: Covering the middle (low frequencies) of the magnitude spectrum, resulting a high pass filter
    Right image: Removing low frequency parts (blurry background), plotted in JET [1]
 
 <img src="/demo/1mag_orig.png" alt="description" height="175" width="300" /> <img src="/demo/2mag_cube.png" alt="description" height="175" width="300" /> <img src="/demo/3rem.png" alt="description" height="175" width="300" />
 
 
 
 
 
 [1] OpenCV Documentation, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
 
 
