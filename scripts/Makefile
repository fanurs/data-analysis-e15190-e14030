GXX = g++
CXX_FLAGS = -fconcepts
CXX_FLAGS := -fPIC $(CXX_FLAGS) # path-independent code
CXX_FLAGS := -O2 $(CXX_FLAGS) # optimization
CXX_FLAGS := `root-config --cflags --libs` $(CXX_FLAGS) # for ROOT; already contained <nlohmann/json.hpp>

calibrate:
	$(GXX) calibrate.cpp src/*.cpp -o calibrate.exe -std=c++20 $(CXX_FLAGS) -I./include -lMathMore -w

remove_tclass:
	$(GXX) remove_tclass.cpp -o remove_tclass.exe -std=c++17 $(CXX_FLAGS) -w

geo_efficiency:
	$(GXX) geo_efficiency.cpp -o geo_efficiency.exe -std=c++20  $(CXX_FLAGS)
