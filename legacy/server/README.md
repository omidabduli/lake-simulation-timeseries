# Legacy server deployment

The application is now served as a static site from **GitHub Pages** (see the
main [README](../../README.md) and
[MIGRATION_TO_GITHUB_PAGES.md](../../MIGRATION_TO_GITHUB_PAGES.md)). The files
in this folder belong to the previous server-based setup. They are kept, still
working, so the application can be self-hosted or rolled back to the Hetzner
server if ever needed.

| File | Purpose |
| --- | --- |
| `Dockerfile`, `docker-compose.yml`, `Dockerfile.dockerignore` | Run the Streamlit server in a container (any Docker host). |
| `deploy.command` | One-click SSH deployment to the Hetzner/CloudPanel server. Kept **locally only** (git-ignored via `.git/info/exclude`). It contains no credentials; it reads them from `.env`. |
| `.env.example` | The variables `deploy.command` expects in the repository-root `.env`. |

## How the Hetzner deployment worked

`deploy.command` connected to the server over SSH with a password
(`sshpass`), synced the repository to `/home/<SITE_USER>/htdocs/<SITE_DOMAIN>`
with `rsync`, created a Python virtual environment there, installed
`requirements.txt` and started Streamlit in the background:

```bash
nohup venv/bin/streamlit run app.py --server.port=<SITE_PORT> --server.address=0.0.0.0 \
  --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false &
```

CloudPanel's nginx reverse proxy served `https://<SITE_DOMAIN>` from that port.
There was no systemd unit, so the process did not survive a server reboot.

## Rolling back to the Hetzner server

1. Make sure the server and its CloudPanel site for the domain still exist.
2. Put the connection settings in the repository-root `.env` (see `.env.example`).
3. Run `legacy/server/deploy.command` (needs `sshpass` and `rsync`).
4. In your DNS zone, point the `time-series` host back to the server:
   replace the `CNAME` record for GitHub Pages with the previous `A` record.
5. In the GitHub repository, open **Settings → Pages** and remove the custom
   domain (or unpublish the site), so GitHub stops claiming the hostname.

`app.py` still runs unchanged on a normal Streamlit server; the browser-specific
code paths are only active under Pyodide.

## Running the container

From the repository root:

```bash
docker compose -f legacy/server/docker-compose.yml up --build -d
```

Then open <http://localhost:8501>. Stop it with
`docker compose -f legacy/server/docker-compose.yml down`.

Without Compose:

```bash
docker build -f legacy/server/Dockerfile -t lake-time-series-forecasting .
docker run -d -p 8501:8501 -v "$(pwd)/logs:/app/logs" --name lake-forecasting lake-time-series-forecasting
```

## Security notes

- Rotate the server password: an earlier version of `deploy.command` with the
  password hard-coded was committed to the public repository history.
- Prefer SSH keys over `sshpass` password logins, and stop disabling host-key
  checking (`StrictHostKeyChecking=no`) if this script is used again.
