
NVCC    = nvcc
OBJ     = main.o kernel.o
EXE     = vecmax


default: $(EXE)

%.o: %.cu
	$(NVCC) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf $(OBJ) $(EXE)

