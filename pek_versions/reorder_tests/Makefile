CC	= gcc
INC	= -I$(AMDAPPSDKROOT)/include -I$(AMDAPPSDKROOT)/include/CAL
LIBS	= -lOpenCL -lm -L$(AMDAPPSDKROOT)/lib/x86_64/

OPTIMIZE= -Wall -O4 -std=gnu99 -msse3 -ggdb
CFLAGS	= $(OPTIMIZE) $(INC)

OBJS	= reorder_wrapper.o
PROGRAM	= reorder_test

$(PROGRAM) : $(OBJS)
	$(CC) $(OBJS) $(LIBS) -o $@

clean:
	rm -rf *.o *~ $(PROGRAM)

.c.o:
	$(CC) $(CFLAGS) -c $<

main_wrapper.o :  reorder_wrapper.c
