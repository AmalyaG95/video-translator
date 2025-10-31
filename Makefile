.PHONY: setup run stop clean test docker-build docker-up docker-down docker-logs help

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
NC     := \033[0m # No Color

help: ## Show this help message
	@echo '${GREEN}Video Translator - Hybrid NestJS + Python ML${NC}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${NC} <target>'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${YELLOW}%-15s${NC} %s\n", $$1, $$2}'

setup: ## Setup all services (install dependencies)
	@echo "${GREEN}Setting up Video Translator...${NC}"
	@echo "${YELLOW}Setting up Python ML service...${NC}"
	cd backend-python-ml && python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	@echo "${YELLOW}Setting up NestJS API...${NC}"
	cd backend-nestjs && npm install
	@echo "${YELLOW}Setting up Frontend...${NC}"
	cd frontend && npm install
	@echo "${GREEN}Setup complete!${NC}"

run: ## Run all services in development mode
	@echo "${GREEN}Starting Video Translator (Development)...${NC}"
	@echo "This will start 3 services:"
	@echo "  - Python ML Service (port 50051)"
	@echo "  - NestJS API Gateway (port 3001)"
	@echo "  - Next.js Frontend (port 3000)"
	@echo ""
	@echo "${YELLOW}Note: Run each in a separate terminal:${NC}"
	@echo "  Terminal 1: cd backend-python-ml && . venv/bin/activate && python src/main.py"
	@echo "  Terminal 2: cd backend-nestjs && npm run start:dev"
	@echo "  Terminal 3: cd frontend && npm run dev"

docker-build: ## Build Docker images
	@echo "${GREEN}Building Docker images...${NC}"
	@./docker-with-permissions.sh build

docker-up: ## Start services with Docker Compose
	@echo "${GREEN}Starting Video Translator with Docker Compose...${NC}"
	@./docker-with-permissions.sh up -d
	@echo "${GREEN}Services started!${NC}"
	@echo "Frontend: http://localhost:3000"
	@echo "API: http://localhost:3001"
	@echo "ML Service: localhost:50051 (gRPC)"

docker-down: ## Stop Docker services
	@echo "${YELLOW}Stopping Docker services...${NC}"
	@./docker-with-permissions.sh down

docker-logs: ## Show Docker logs
	@./docker-with-permissions.sh logs -f

stop: docker-down ## Stop all services

clean: ## Clean up generated files and dependencies
	@echo "${YELLOW}Cleaning up...${NC}"
	@./docker-with-permissions.sh down -v || true
	rm -rf backend-nestjs/node_modules backend-nestjs/dist
	rm -rf frontend/node_modules frontend/.next
	rm -rf backend-python-ml/venv
	rm -rf .data/artifacts/* .data/temp_work/* .data/uploads/* .data/logs/*
	@echo "${GREEN}Cleanup complete!${NC}"

test: ## Run tests
	@echo "${GREEN}Running tests...${NC}"
	@echo "${YELLOW}Testing NestJS API...${NC}"
	cd backend-nestjs && npm test
	@echo "${YELLOW}Testing Python ML service...${NC}"
	cd backend-python-ml && . venv/bin/activate && pytest tests/
	@echo "${GREEN}All tests passed!${NC}"

install-python-deps: ## Install Python ML dependencies
	cd backend-python-ml && . venv/bin/activate && pip install -r requirements.txt

install-nestjs-deps: ## Install NestJS dependencies
	cd backend-nestjs && npm install

install-frontend-deps: ## Install Frontend dependencies
	cd frontend && npm install

dev-python: ## Run Python ML service in development
	cd backend-python-ml && . venv/bin/activate && python src/main.py

dev-nestjs: ## Run NestJS API in development
	cd backend-nestjs && npm run start:dev

dev-frontend: ## Run Frontend in development
	cd frontend && npm run dev

.DEFAULT_GOAL := help
