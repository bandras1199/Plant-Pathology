# Plant-Pathology
Experimenting with the Kaggle Plant Pathology Dataset [1]

Initial approach: <br />

  I wanted to reduce the computational needs as much as possible, using image processing and trying to 
  find a good performing but still sustainable network architecture.
  
<strong> Introduction to the dataset: </strong> <br />
1820 training images with different disease categories: <br />  <br />
          Healthy label and Scab label: <br />
 <img src="/demo/healthy.png" alt="description" height="175" width="300" /> <img src="/demo/scab.png" alt="description" height="175" width="300" />   <br />
           Rust label and Multiple label: <br />
 <img src="/demo/rust.png" alt="description" height="175" width="300" /> <img src="/demo/multiple.png" alt="description" height="175" width="300" />   <br />

<strong> Image processing steps: </strong> 

  1, Initial image:
  
 <img src="/demo/0orig.png" alt="description" height="175" width="300" />   
 
  2, Removing blurry background noise:
  - Left image: Applying Fourier transformation on the image and plotting the magnitude spectrum <br />
  - Middle image: Covering the middle (low frequencies) of the magnitude spectrum, resulting a high pass filter <br />
  - Right image: Removing low frequency parts (blurry background), plotted in JET [2] <br />
 
 <img src="/demo/1mag_orig.png" alt="description" height="175" width="300" /> <img src="/demo/2mag_cube.png" alt="description" height="175" width="300" /> <img src="/demo/3rem.png" alt="description" height="175" width="300" />
 
 3, Object detection:
 - Left image: Sharping the edges then detecting the outlier contours <br />
 - Middle image: First fit an ellipse on the contours of the leaf, make it parallel to X axis and then fit the biggest possible rectangle inside this ellipse by solving the optimization problem [3] <br />
 - Right image: Final image, cropped to standard size <br />
 
  <img src="/demo/4cont.png" alt="description" height="175" width="300" /> <img src="/demo/5fitellipse4.png" alt="description" height="175" width="300" /> <img src="/demo/6compr.png" alt="description" height="175" width="300" />
  
4, Augmentation: 
- Applying random distortion and rotation on the image, automatically crop the black sides [4] [5] <br />
- Image: Previous image after augmentation <br />
 <img src="/demo/7augmentation.png" alt="description" height="175" width="300" />
 
Mistakes: 
- Some images have their infected part close to their edge, which has to be considered during object detection. Currently, these faulty images are excluded from the experiment.
 - Left image: Original image <br />
 - Right image: Processed Image <br />
<img src="/demo/8mistake2.png" alt="description" height="175" width="300" /> <img src="/demo/8mistake.png" alt="description" height="175" width="300" /> 

<strong> Deep Learning: </strong> <br />
I have tried two popular convolutional neural network architecture, ResNet and DenseNet because of their advantages and simple implementation [6] [7].

ResNet50 had so far the best results in a 4-by-4 cross validation split. It had similar training graphs over each CV cycle best performance on the most difficult class (multiple disease). <br />
I made a custom modul ResNet18 to reduce training time although the it ended with worse results, therefore the complexity of the more layers is necessary to solve this problem [8]. <br />

 <img src="/demo/Resnet50.png" alt="description" height="231" width="300" /> <img src="/demo/Dense121.png" alt="description" height="231" width="300" /> <img src="/demo/Res18.png" alt="description" height="231" width="300" />  <br />
 <img src="/demo/plot.png" alt="description" height="420" width="800" /> 
 
 
<strong> Conclusion, future plans: </strong> <br />
  - fix image processing method for leaf injuries placed close to the side
  - focus on the detection of multiple diseases
  - apply histogram based features and spectral filtering
  - select best model during training (not the last) - requires saving which takes extra time
  - add randomized repetition and 10-by-10 CV for more accurate evaluation
 
 <strong> References: </strong> 
 <br /> [1] Kaggle Plant Pathology Challange, https://www.kaggle.com/c/plant-pathology-2020-fgvc7
 <br /> [2] OpenCV Documentation, https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
 <br /> [3] Calculus Optimization #9, https://www.youtube.com/watch?v=r0wdreyN4QE
 <br /> [4] Image Augmentation in Numpy, https://medium.com/@schatty/image-augmentation-in-numpy-the-spell-is-simple-but-quite-unbreakable-e1af57bb50fd
 <br /> [5] Automatic crop after rotation, https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Operations.py 
 <br /> [6] ResNet, https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
 <br /> [7] DenseNet, https://medium.com/the-advantages-of-densenet/the-advantages-of-densenet-98de3019cdac
 <br /> [8] ResNet code, https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/resnet.py
 
 
