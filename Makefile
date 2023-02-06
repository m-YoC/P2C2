.DEFAULT_GOAL := help

This-is: ## <write help text here.>
	:
docker-up: ## docker-compose up -d
	docker-compose up -d
docker-down: ## docker-compose down
	docker-compose down
docker-build: ## docker-compose build
	docker-compose build
docker-modified-cgroup-dir:
	sudo mkdir /sys/fs/cgroup/systemd && sudo mount -t cgroup -o none,name=systemd cgroup /sys/fs/cgroup/systemd
help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
