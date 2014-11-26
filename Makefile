main: main.rs
	rustc $^

run_tests: main.rs
	rustc --test $^ -o $@

test: run_tests
	#RUST_BACKTRACE=1 ./run_tests
	./run_tests

.PHONY: clean
clean:
	$(RM) main run_tests
