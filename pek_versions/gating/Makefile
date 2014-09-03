CC	= gcc
ATISDK	= /opt/AMDAPP
INC	= -I$(ATISDK)/include -I$(ATISDK)/include/CAL
LIBS	= -lOpenCL -L/opt/AMDAPP/lib/x86_64/ -lm

OPTIMIZE= -Wall -O4 -std=gnu99 -msse3 -DSFMT_MEXP=19937 -ggdb
CFLAGS	= $(OPTIMIZE) $(INC)
#SRCS	= main_multi-freq-2.c SFMT-src-1.4/SFMT.c

#step 1: (gets the Mersenne Twister .o file in the right place
#OBJS	= ./SFMT-src-1.4/SFMT.o SFMT.o main_multi-freq-4interleaved_output_xor_gating.o #32chanwrap.o ./SFMT-src-1.4/SFMT.o

#step 2: (to compile the program)
#OBJS	= SFMT.o main_multi-freq-4interleaved_output_xor_gating_for16.o
#OBJS	= SFMT.o main_multi-freq-4interleaved_output_xor_multidataset_alternate.o
OBJS	= SFMT.o main_multi-freq-4interleaved_output_xor_gating.o
PROGRAM	= full_gating_run

$(PROGRAM) : $(OBJS)
	$(CC) $(OBJS) $(LIBS) -o $@

clean:
	rm -rf *.o *~ $(PROGRAM) *.img *dis.txt #kernel*.h

.c.o:
	$(CC) $(CFLAGS) -c $<

SFMT.o : ./SFMT-src-1.4/SFMT.c

#main_multi-freq-4interleaved_output_xor_gating_for16.o :  main_multi-freq-4interleaved_output_xor_gating_for16.c #kernel.h kernel2.h
#main_multi-freq-4interleaved_output_xor_multidataset_alternate.o :  main_multi-freq-4interleaved_output_xor_multidataset_alternate.c #kernel.h kernel2.h
main_multi-freq-4interleaved_output_xor_gating.o :  main_multi-freq-4interleaved_output_xor_gating.c #kernel.h kernel2.h


#Note I don't know how to use make files very well... after a clean, to make the SFMT.o, it needs ./SFMT-src-1.4/SFMT.o in OBJS before it decides to make the object file
#So run like that, then run again without, once the .o file is made
