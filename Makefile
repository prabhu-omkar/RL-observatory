# RL Observatory — Makefile
# ─────────────────────────────────────────────────────────────────────────────
# Common workflows as one-liners.  Run `make help` to see all targets.
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
SHELL         := /bin/bash
PYTHON        := python3
VENV          := venv
PIP           := $(VENV)/bin/pip
ACTIVATE      := source $(VENV)/bin/activate

# ── Colours ──────────────────────────────────────────────────────────────────
CYAN  := \033[96m
GREEN := \033[92m
RESET := \033[0m

.PHONY: help install backend agent stop clean lint

help: ## Show this help message
	@echo ""
	@echo "  $(CYAN)RL Observatory$(RESET) — Available targets"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-14s$(RESET) %s\n", $$1, $$2}'
	@echo ""

install: ## Create venv & install Python dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed.$(RESET)"

backend: ## Start SigNoz backend via Foundry
	foundry cast apply casting.yaml
	@echo "$(GREEN)✓ SigNoz backend is starting — UI at http://localhost:3301$(RESET)"

agent: ## Run the RL Observatory agent (ensure Unity is in Play mode)
	$(ACTIVATE) && $(PYTHON) run_agent.py

stop: ## Tear down the SigNoz backend
	foundry cast down casting.yaml
	@echo "$(GREEN)✓ Backend stopped.$(RESET)"

clean: ## Remove venv and Python caches
	rm -rf $(VENV) __pycache__ *.egg-info .eggs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned.$(RESET)"

lint: ## Run basic Python linting
	$(ACTIVATE) && $(PYTHON) -m py_compile run_agent.py
	@echo "$(GREEN)✓ Syntax OK.$(RESET)"
