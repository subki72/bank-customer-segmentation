# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, reproduction steps, and estimated severity.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env`, `.env.*`, and private key files are excluded via `.gitignore`; zero hardcoded credentials exist in source control.
- **Input Validation**: All API request bodies are validated with Pydantic v2 bounds (`Usia`, `Jumlah_Anak`, `Pendapatan_Tahunan`, `Total_Relationship_Balance`).
- **Safe Error Boundaries**: Centralized FastAPI exception handlers log diagnostics internally without exposing stack traces externally.
- **Automated Scanning**: Weekly Dependabot security updates (`semver-major` ignored), Semgrep SAST, and Trivy container vulnerability scanning in CI.
- **Container Hardening**: Multi-stage non-root Docker builds running as `appuser` (`UID 10001`).
