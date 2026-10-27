# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private
vulnerability reporting instead: the repository's **Security** tab → **Report a vulnerability**.
Include what you found, how to reproduce it and what an attacker could do with it. You should
get a reply within a week.

Only the latest commit on the default branch is supported; there are no maintained release
branches.

## What is in scope

BioSphere is meant to be self-hosted. The areas where a bug matters most:

- **Authentication and ownership** — sessions (`engine/biosphere/api/deps.py`), the
  property-scoped checks in the `require_*` helpers, and QR **capture links**, which act as the
  property's owner but only on an allow-list of capture routes (`_CAPTURE_ROUTES`).
- **Public endpoints** under `/api/v1/public/**` — embed reads, measurement, and the
  investment-assistant chat, which spends money on a paid API and is rate-limited and
  budget-capped.
- **Uploads** — size and type checks, ffprobe sniffing, chunked staging.

## Things that are deliberate, not vulnerabilities

- The development defaults: CORS allows `localhost:3000`, the engine binds to `127.0.0.1`, and
  the LAN proxy uses a self-signed certificate you generate yourself.
- The session token is kept in `localStorage` and sent as a header (the engine also sets an
  httpOnly cookie for same-origin deployments). See the note at the top of `web/src/lib/session.ts`.
- With an API key configured, the listing-copy and investment-assistant features send property
  facts and visitors' questions to Anthropic. They are off until a key is set.

## Running a public instance

Set `BIOSPHERE_ENV=production`, tighten `BIOSPHERE_CORS_ALLOW_ORIGINS`, put the engine behind
HTTPS, consider `BIOSPHERE_ALLOW_REGISTRATION=false`, and review the investment assistant's
budget caps before switching it on.
