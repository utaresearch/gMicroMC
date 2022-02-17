NVCC=nvcc
CFLAGS=-I ./inc -std=c++14 -dlto -w

ODIR=obj
SDIR=src
PROGRAM=gMicroMC

gMicroMC : $(ODIR)/*.o
	$(NVCC) $^ -o $@ 

$(ODIR)/*.o: $(SDIR)/*
	$(NVCC) main.cpp $^ -dc $(CFLAGS) 
	mv *.o $(ODIR)

main : 
	$(NVCC) main.cpp -dc $(CFLAGS)
	mv *.o $(ODIR)
	$(NVCC) $(ODIR)/*.o -o $(PROGRAM)

chem : 
	$(NVCC) $(SDIR)/chemical.cpp -dc $(CFLAGS)
	mv *.o $(ODIR)
	$(NVCC) $(ODIR)/*.o -o $(PROGRAM)
phys : 
	$(NVCC) $(SDIR)/physics* -dc $(CFLAGS)
	mv *.o $(ODIR)
	$(NVCC) $(ODIR)/*.o -o $(PROGRAM)
clean:
	rm $(ODIR)/*
	rm gMicroMC
