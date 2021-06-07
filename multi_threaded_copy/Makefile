OBJS		= main.o multi_threaded_copy.o
SOURCE		= main.cpp multi_threaded_copy.cu
HEADER		= constants.h multi_threaded_copy.h
OUT			= my_jacobi.out
CC	 		= g++
FLAGS		= -g -c -Wall -fopenmp
LFLAGS_OCV	= -L/apps/2021/cuda/11.0/lib64 -L/apps/2021/opencv/4.5.1/lib64 -I/apps/2021/opencv/4.5.1/include -lcuda -lcudart -fopenmp -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_stitching -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core
LFLAGS 		= -L/apps/2021/cuda/11.0/lib64 -lcuda -lcudart -fopenmp
NVCC		= nvcc
NVCC_FLAGS 	= -Xptxas --optimize-float-atomics -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt
WIDTH	   ?= 8192
HEIGHT	   ?= 8192

all: $(OBJS)
ifdef write_images
	@module load opencv/4.5.1 gcc/10.2.0 cuda/11.0; \
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS_OCV)
else
	@module load gcc/10.2.0; \
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)
endif

main.o: main.cpp
ifdef write_images
	@module load opencv/4.5.1 gcc/10.2.0; \
	$(CC) $(FLAGS) -D WRITE_IMG -D WIDTH=$(WIDTH) -D HEIGHT=$(HEIGHT) main.cpp -std=c++11
else
	$(CC) $(FLAGS) main.cpp -std=c++11
endif

multi_threaded_copy.o: multi_threaded_copy.cu
ifdef write_images
	@module load cuda/11.0; \
	$(NVCC) $(NVCC_FLAGS) -D WIDTH=$(WIDTH) -D HEIGHT=$(HEIGHT) -c multi_threaded_copy.cu -std=c++11
else
	@module load cuda/11.0; \
	$(NVCC) $(NVCC_FLAGS) -D WIDTH=$(WIDTH) -D HEIGHT=$(HEIGHT) -c multi_threaded_copy.cu -std=c++11
endif

clean:
	rm -f $(OBJS) $(OUT)

run: clean all
ifdef write_images
ifdef gpus
		@module load opencv/4.5.1 cuda/11.0; \
		./$(OUT) -DWRITE_IMG $(gpus)
else
		@module load opencv/4.5.1 cuda/11.0; \
		./$(OUT)
endif
else
ifdef gpus
		@module load cuda/11.0; \
		./$(OUT) $(gpus)
else
		@module load cuda/11.0; \
		./$(OUT)
endif
endif

bench:
ifdef gpus
	@module load cuda/11.0; \
	nvprof ./$(OUT) $(gpus)
else
	@module load cuda/11.0; \
	nvprof ./$(OUT)
endif