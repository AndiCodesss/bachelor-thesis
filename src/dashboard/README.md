# Dashboard

The dashboard is a local operator UI for the research runtime. It covers four workflows:

- cache rebuilds
- autonomy launch and monitoring
- cleanup runs
- signal inspection

## Backend

Run the API from the repository root:

```bash
uv run --with fastapi --with uvicorn --with websockets --with pydantic \
  uvicorn src.dashboard.backend.main:app --port 8000
```

## Frontend

Run the Vite app from `src/dashboard/frontend`:

```bash
npm run dev
```

Optional environment variables:

- `VITE_API_ORIGIN`: defaults to `http://localhost:8000`
- `VITE_WS_ORIGIN`: defaults to the websocket form of `VITE_API_ORIGIN`

## One-command Start

From the repo root:

```bash
./start-dashboard.sh
```

The UI is available at `http://localhost:5173`.
