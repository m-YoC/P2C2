.DEFAULT_GOAL := help

This-is: ## <write help text here.>
	:
run: ## run with headless environment
	xvfb-run python main.py
refresh-res: ## delete all files in result directory
	rm ./result/* && touch .keep
help: ## Display this help screen
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
