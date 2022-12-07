GXX = g++
CXX_FLAGS = -std=c++20
CXX_FLAGS := -fconcepts $(CXX_FLAGS)
CXX_FLAGS := -fPIC $(CXX_FLAGS) # path-independent code
CXX_FLAGS := -O2 $(CXX_FLAGS) # optimization
CXX_FLAGS := `root-config --cflags --libs` $(CXX_FLAGS) # for ROOT; already contained <nlohmann/json.hpp>

target:
	$(GXX) calibrate.cpp src/*.cpp -o calibrate.exe $(CXX_FLAGS) -I./include -lMathMore -w
	$(GXX) remove_tclass.cpp -o remove_tclass.exe $(CXX_FLAGS)
