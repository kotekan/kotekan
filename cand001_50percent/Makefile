CC	= gcc
ATISDK	= /opt/AMDAPP
INC	= -I$(ATISDK)/include -I$(ATISDK)/include/CAL
LIBS	= -lOpenCL -L/opt/AMDAPP/lib/x86_64/ -lm

OPTIMIZE= -Wall -O4 -std=c99 -msse3
CFLAGS	= $(OPTIMIZE) $(INC)

OBJS	= main.o
PROGRAM	= run

$(PROGRAM) : $(OBJS) kernel.h
	$(CC) $(OBJS) $(LIBS) -o $@

clean:
	rm -rf *.o *~ $(PROGRAM) *.img *dis.txt kernel*.h

.c.o:
	$(CC) $(CFLAGS) -c $<

main.o : main.c kernel.h


kernel.h : test.cl
	./template-converter kern $< >| $@
