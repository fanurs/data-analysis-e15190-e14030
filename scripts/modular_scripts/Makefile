GXX = g++
CXX_FLAGS = -std=c++20
CXX_FLAGS := -fconcepts $(CXX_FLAGS)
CXX_FLAGS := -fPIC $(CXX_FLAGS)
CXX_FLAGS := -O2 $(CXX_FLAGS)
CXX_FLAGS := `root-config --cflags --libs` $(CXX_FLAGS)

target:
	$(GXX) $(SCRIPT) -o $(basename $(SCRIPT)).exe $(CXX_FLAGS)
