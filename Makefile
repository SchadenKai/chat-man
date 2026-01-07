run-dev-backend:
	cd backend && uv run fastapi dev ./app/main.py

run-dev-frontend:
	cd frontend && pnpm run dev

run-containers:
	cd deployment && docker-compose up -d --build