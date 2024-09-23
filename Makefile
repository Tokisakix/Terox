build_terox:
	mkdir -p build/
	cd build/ && cmake .. && make -j 4

build:
	make build_terox
	mv build/cpp_function.* terox/_C/
	mv build/cuda_function.* terox/_C/

dev:
	python -m pip install -Ue .

test:
	pytest

clean:
	rm -rf build/
	rm -rf terox/_C/*.so
	rm -rf *.egg-info
	rm -rf .pytest_cache