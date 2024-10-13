CC = gcc
CFLAGS = -fopenmp -I/usr/include/cfitsio
LDFLAGS = -lcfitsio -lm

lab1: lab1.o
	$(CC) $(CFLAGS) lab1.o -o lab1 $(LDFLAGS)

lab1.o: lab1.c
	$(CC) $(CFLAGS) -c lab1.c

clean:
	rm -f lab1 lab1.o
