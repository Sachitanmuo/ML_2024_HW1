CC=g++
CFLAGS=-std=c++11
SRC=HW1.cpp Model.cpp
EXECUTABLE=HW1

all: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
