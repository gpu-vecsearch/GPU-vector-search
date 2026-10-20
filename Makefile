.PHONY: v1 v2 v3 v4 v5 v6 bench clean

v1 v2 v3 v4 v5 v6 bench:
	@echo "$@ is not implemented yet; add its build or benchmark recipe to the Makefile."
	@exit 1

clean:
	rm -rf build