FaceX
=====

A high performance open source face landmarks detector (face alignment), based on explicit shape regression algorithm.

There are some open source face landmarks detectors, but most of them are only for research purpose, and only offer MATLAB code. This face landmarks detector is coded in C++, with performance and modularity in mind.

The algorithm used is based on [1]. And the model file (model.xml.gz) is trained on LFPW face database, with annotations offered by ibug group [2]. Thanks to these guys.

This detector is coded in Visual C++ 2013 Express for Desktop, and uses OpenCV. I believe it will work on other compilers with minor or even no modifications. The most important thing is linking the OpenCV library correctly. Currently it use these modules: core, highgui, imgproc, objdetect. Also notice that it uses some C++ 11 features, so be sure your compiler is up to date.

Currently, the training code is not available. I will make it available as soon as I make the training code clean enough.

Known Issue:

1. FaceX::OpenModel is very slow (takes more than 40 seconds) because the persistence module of OpenCV is slow. Maybe I will consider using binary file to store the model.


[1] Cao X, Wei Y, Wen F, et al. Face alignment by explicit shape regression[J]. International Journal of Computer Vision, 2014, 107(2): 177-190.
[2] http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
