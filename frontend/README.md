# Medgent Frontend

Independent frontend app (Vite + React + TypeScript), migrated from `/ui` static page.

## Run

```bash
cd frontend
npm install
npm run dev
```

Open: `http://127.0.0.1:5173`

## Environment

Copy and edit:

```bash
cp .env.example .env
```

- `VITE_API_BASE` default: `http://127.0.0.1:8000/api/v1`
- `VITE_API_KEY` default: `dev-local-key`

## Build

```bash
npm run build
npm run preview
```
