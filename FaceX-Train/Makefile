CFLAGS = -std=c++11 -O3

objects = main_train.o fern_train.o regressor_train.o utils_train.o

facex-train: $(objects)
	g++ $(CFLAGS) -o facex-train $(objects) -lopencv_core -lopencv_highgui

main_train.o: main_train.cpp regressor_train.h fern_train.h utils_train.h
	g++ $(CFLAGS) -c main_train.cpp

fern_train.o: fern_train.cpp fern_train.h utils_train.h
	g++ $(CFLAGS) -c fern_train.cpp

regressor_train.o: regressor_train.cpp regressor_train.h fern_train.h utils_train.h
	g++ $(CFLAGS) -c regressor_train.cpp

utils_train.o: utils_train.cpp utils_train.h
	g++ $(CFLAGS) -c utils_train.cpp 

clean:
	-rm facex-train $(objects)
