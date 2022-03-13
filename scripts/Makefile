GXX = g++
CXX_FLAGS = -std=c++17
CXX_FLAGS := -fconcepts $(CXX_FLAGS)
CXX_FLAGS := -fPIC $(CXX_FLAGS) # path-independent code
CXX_FLAGS := -O2 $(CXX_FLAGS) # optimization
CXX_FLAGS := `root-config --cflags --libs` $(CXX_FLAGS) # for ROOT
CXX_FLAGS := -lMathMore $(CXX_FLAGS) # needed for ROOT::Math
CXX_FLAGS := -I./include $(CXX_FLAGS) # custom headers

target:
	$(GXX) calibrate.cpp src/*.cpp -o calibrate.exe $(CXX_FLAGS)
