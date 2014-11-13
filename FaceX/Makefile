CFLAGS = -std=c++11 -O3
objects = main.o face_x.o fern.o regressor.o utils.o

facex: $(objects)
	g++ $(CFLAGS) -o facex $(objects) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
	@echo To run the demo program, make sure haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the working directory.

main.o: main.cpp face_x.h regressor.h fern.h utils.h
	g++ $(CFLAGS) -c main.cpp

face_x.o: face_x.cpp face_x.h regressor.h fern.h utils.h
	g++ $(CFLAGS) -c face_x.cpp

fern.o: fern.cpp fern.h utils.h
	g++ $(CFLAGS) -c fern.cpp

regressor.o: regressor.cpp regressor.h fern.h utils.h
	g++ $(CFLAGS) -c regressor.cpp

utils.o: utils.cpp utils.h
	g++ $(CFLAGS) -c utils.cpp 

clean:
	-rm facex $(objects)	
