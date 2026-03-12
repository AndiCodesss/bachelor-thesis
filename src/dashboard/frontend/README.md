# Frontend

This is the Vite + React operator UI for the local research dashboard.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Configuration

The app talks to the local dashboard backend through Vite env vars:

- `VITE_API_ORIGIN`
- `VITE_WS_ORIGIN`

If unset, the defaults are `http://localhost:8000` and `ws://localhost:8000`.
