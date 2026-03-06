# NQ-Alpha Cache Runner Dashboard

A modern, glassmorphism-themed web dashboard to cleanly set constraints, execute, and monitor the `cache_runner.py` real-time progress.

## Technologies Used

- **Backend**: FastAPI + WebSockets (Python). Isolated via `uv run --with ...` so it doesn't pollute your main project dependencies.
- **Frontend**: Vite + React + TypeScript.
- **Styling**: Pure CSS with rich dark-mode aesthetics, dynamic background blob animations, glassmorphism panels, and vibrant accents.

## Running the Dashboard

You will need two terminal windows to run the dashboard. Please execute these from the **root of your bachelor project**.

### 1. Start the Backend API

In your first terminal, from the root where `pyproject.toml` is located, run:

```powershell
uv run --with fastapi --with uvicorn --with websockets --with pydantic uvicorn src.dashboard.backend.main:app --port 8000
```

### 2. Start the Frontend UI

In your second terminal, launch the Vite dev server:

```powershell
cd src/dashboard/frontend
npm run dev
```

### 3. Open the Dashboard

Navigate to [http://localhost:5173](http://localhost:5173) in your browser. Configure your runner execution, and hit the start button. The terminal window will stream live logs directly from the background cache runner task!
