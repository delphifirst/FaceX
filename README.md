FaceX
=====

A high performance (real-time) open source face landmarks detector (face alignment), based on explicit shape regression algorithm. Training code is also provided.

There are some open source face landmarks detectors, but most of them are only for research purpose, and only offer MATLAB code. This face landmarks detector is coded in C++, with performance and modularity in mind.

The algorithm used is based on [1]. model.xml.gz is trained on LFPW, HELEN, AFW and IBUG face databases, with annotations offered by ibug group [2]. It offers 51 landmarks. model_small.xml.gz is trained on face databases provided by [3]. It offers 5 landmarks with higher speed. Thanks to these guys.

The license of the face landmarks detector (code in FaceX directory) is MIT. And the license of the training tool (code in FaceX-Train directory) is GPL v3.

Notice
====

This detector is coded in Visual C++ 2013 Express for Desktop, and uses OpenCV. It also works in GCC. I believe it will work on other compilers with minor or even no modifications. The most important thing is linking the OpenCV library correctly. Currently it use these modules: core, highgui, imgproc, objdetect. Also notice that it uses some C++ 11 features, so be sure your compiler is up to date.

When you try the detector code, make sure the three file haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the current working directory.

When you try the training code, I suggest you compile FaceX-Train as 64bit code, since it may use large amount of memory (it will load all the images into memory).

Currently, the training code is a little messy. I hope I can clean it up someday.

How To Train
====

First collect face images with face area and landmark labels. I recommend you first download dataset from [3] to see if the works correctly. You can check FaceX-Train/train to know how to organize the training data. The name of the label file must be labels.txt, and the format of labels.txt is like this:

> image1.png FACE-LEFT FACE-RIGHT FACE-TOP FACE-BOTTOM X1 Y1 X2 Y2 ...

> image2.png ...

Notice that FACE-LEFT, FACE-RIGHT, FACE-TOP, FACE-BOTTOM must be integers and the face region is inclusively constructed (i.e. boarders are included). X1 Y1, X2 Y2, ... can be floating numbers.

Then create a config file for training, you can use FaceX-Train/sample_config.txt as a start point. After that, run command:

> FaceX-Train config.txt

It will take several minutes to several hours, depending on the training-set size and the speed of your computer.

I have already put one image with labels.txt there. It is used to show how to organize those files. It is NOT possible to train a model with just one image.

Known Issue
====

1. The program is very slow if you run it in Visual C++ debugger. Even if you use Release Mode. Therefore, run it directly outside (remember to put the three files in the current working directory). It seems Visual C++ debugger will slow down some program greatly.
2. On some version of Linux, If you use more than one FaceX object in the program, it will consume about 600MB more memory. I don't know the reason, but use tcmalloc to replace the memory allocator of g++ will solve this problem. Maybe there is a bug in OpenCV or g++ memory allocator.
3. One user reported that the training code doesn't work on Mac. I'll check it when I have opportunity. The alignment code works fine however.

Reference
====

[1] Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.

[2] http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

[3] http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
