make all:
	make -C v64
	make -C v61
	make -C v44
	make -C v01
	make -C v21

vXXX:
	make -C vXXX

v64:
	make -C v64

v61:
	make -C v61

v44:
	make -C v44
	
v01:
	make -C v01

v21:
	make -C v21

clean:
	make -C v64 clean
	make -C v61 clean
	make -C v44 clean
	make -C v01 clean
	make -C v21 clean
	make -C vXXX clean

.PHONY: clean vXXX clean v64 clean v61 clean v44 clean v01 clean v21
