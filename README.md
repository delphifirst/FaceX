FaceX
=====

A high performance (real-time) open source face landmarks detector (face alignment), based on explicit shape regression algorithm.

There are some open source face landmarks detectors, but most of them are only for research purpose, and only offer MATLAB code. This face landmarks detector is coded in C++, with performance and modularity in mind.

The algorithm used is based on [1]. model.xml.gz is trained on LFPW, HELEN, AFW and IBUG face databases, with annotations offered by ibug group [2]. It offers 51 landmarks. model_small.xml.gz is trained on face databases provided by [3]. It offers 5 landmarks with higher speed. Thanks to these guys.

Notice
====

This detector is coded in Visual C++ 2013 Express for Desktop, and uses OpenCV. It also works in GCC. I believe it will work on other compilers with minor or even no modifications. The most important thing is linking the OpenCV library correctly. Currently it use these modules: core, highgui, imgproc, objdetect. Also notice that it uses some C++ 11 features, so be sure your compiler is up to date.

When you try this code, make sure the three file haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the current working directory.

Currently, the training code is not available. I will make it available as soon as I make the training code clean enough.

Known Issue
====

1. The program is very slow if you run it in Visual C++ debugger. Even if you use Release Mode. Therefore, run it directly outside (remember to put the three files in the current working directory). It seems Visual C++ debugger will slow down some program greatly.

Reference
====

[1] Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.

[2] http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

[3] http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
