make all:
	make -C us1
	make -C us2
	make -C us3
	make -C v401
	make -C v407
	make -C v500
	make -C v504
	make -C v601
	make -C v602
	make -C v606
	make -C v703
	make -C v704

vXXX:
	make -C vXXX

v504:
	make -C v504

v500:
	make -C v500

v401:
	make -C v401

v407:
	make -C v407

v400:
	make -C v400

v703:
	make -C v703

v704:
	make -C v704

us3:
	make -C us3

us1:
	make -C us1

v606:
	make -C v606

v601:
	make -C v601

us2:
	make -C us2

v602:
	make -C v602


clean:
	make -C us1  clean
	make -C us2  clean
	make -C us3  clean
	make -C v401 clean
	make -C v407 clean
	make -C v500 clean
	make -C v504 clean
	make -C v601 clean
	make -C v602 clean
	make -C v606 clean
	make -C v703 clean
	make -C v704 clean

.PHONY: clean vXXX clean v504 clean v500 clean v401 clean v407 clean v400 clean v703 clean v704 clean us3 clean us1 clean v606 clean v601 clean us2 clean v602
