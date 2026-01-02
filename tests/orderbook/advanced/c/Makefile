# Makefile for C unit tests using µnit
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -I../../../../src/mm_toolbox/orderbook/advanced/c
SRC_DIR = ../../../../src/mm_toolbox/orderbook/advanced/c

SRCS = munit.c test_helpers.c test_ladder.c test_runner.c \
       $(SRC_DIR)/orderbook_helpers.c $(SRC_DIR)/orderbook_ladder.c

test: $(SRCS)
	$(CC) $(CFLAGS) -o test_runner $(SRCS) -lm
	./test_runner

clean:
	rm -f test_runner

.PHONY: test clean

