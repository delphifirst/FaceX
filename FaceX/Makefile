objects = main.o face_x.o fern.o regressor.o utils.o

facex: $(objects)
	g++ -std=c++11 -o facex $(objects) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect
	@echo To run the demo program, make sure haarcascade_frontalface_alt2.xml model.xml.gz test.jpg are in the working directory.

main.o: main.cpp face_x.h regressor.h fern.h utils.h
	g++ -std=c++11 -c main.cpp

face_x.o: face_x.cpp face_x.h regressor.h fern.h utils.h
	g++ -std=c++11 -c face_x.cpp

fern.o: fern.cpp fern.h utils.h
	g++ -std=c++11 -c fern.cpp

regressor.o: regressor.cpp regressor.h fern.h utils.h
	g++ -std=c++11 -c regressor.cpp

utils.o: utils.cpp utils.h
	g++ -std=c++11 -c utils.cpp 

clean:
	-rm facex $(objects)	
