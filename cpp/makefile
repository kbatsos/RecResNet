#Set the compiler

CC = g++

#Set compile time flas

CFLAGS= -fopenmp -shared -O3 -Wl,--export-dynamic

# location of the Python header files

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib

# compile mesh classes
TARGET = cpputils

$(TARGET).so: $(TARGET).o
	$(CC) $(CFLAGS) $(TARGET).o -L$(BOOST_LIB) -lboost_python -lpng16 -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(TARGET).o: $(TARGET).cpp
	$(CC) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -fopenmp -c $(TARGET).cpp

clean:
	rm -rf *.so *.o


