run-dev-backend:
	cd backend && uv run fastapi dev ./app/main.py

run-dev-frontend:
	cd frontend && npm run dev