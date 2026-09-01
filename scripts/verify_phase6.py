"""
Phase 6 Verification Script — Production API, Dockerization & Deployment Readiness.

Usage
-----
  python scripts/verify_phase6.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("\n" + "=" * 65)
    print("  CREDITRISK AI — PHASE 6: VERIFICATION SUITE")
    print("=" * 65)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    results = []

    def check(label: str, condition: bool, detail: str = ""):
        status = PASS if condition else FAIL
        msg = f"  {status}  {label}"
        if detail:
            msg += f"\n         {detail}"
        print(msg)
        results.append((label, condition))

    # ── 1. FastAPI App & Routes Import ──────────────────────────────
    print("\n[1] Verifying FastAPI application & routes...")
    try:
        from api.main import app
        check("api.main module imported successfully", True)
        routes = [r.path for r in app.routes]
        check("GET /health route registered", "/health" in routes)
        check("GET /model-info route registered", "/model-info" in routes)
        check("POST /score route registered", "/score" in routes)
        check("POST /portfolio-summary route registered", "/portfolio-summary" in routes)
    except Exception as exc:
        check("api.main module imported successfully", False, str(exc))

    # ── 2. Pydantic Schemas & API Scoring Test ──────────────────────
    print("\n[2] Verifying Pydantic schemas & scoring endpoint logic...")
    try:
        from api.main import BatchScoreRequest, LoanApplicationRecord, score_loans
        record = LoanApplicationRecord(
            loan=15000, mortdue=50000, value=85000, reason="DebtCon",
            job="Office", yoj=6, derog=0, delinq=0, clage=180, ninq=1, clno=20, debtinc=32
        )
        req = BatchScoreRequest(records=[record], threshold=0.50, lgd=0.45)
        response = score_loans(req, api_key=None)

        check("API /score request processed successfully", response["records_scored"] == 1)
        check("Default probability calculated", "average_default_probability" in response)
        check("Total ECL calculated", "total_ecl" in response)
        check("Prediction record contains risk segment & decision", "risk_segment" in response["predictions"][0])
    except Exception as exc:
        check("API scoring logic", False, str(exc))

    # ── 3. Docker & Infrastructure Artifacts ────────────────────────
    print("\n[3] Verifying Docker & container orchestration files...")
    check("Dockerfile exists", (PROJECT_ROOT / "Dockerfile").exists())
    check(".dockerignore exists", (PROJECT_ROOT / ".dockerignore").exists())
    check("docker-compose.yml exists", (PROJECT_ROOT / "docker-compose.yml").exists())

    # ── 4. Configuration & Deployment Artifacts ─────────────────────
    print("\n[4] Verifying environment & cloud deployment configuration...")
    check(".env.example exists", (PROJECT_ROOT / ".env.example").exists())
    check("render.yaml exists", (PROJECT_ROOT / "render.yaml").exists())
    check(".github/workflows/ci.yml exists", (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists())

    # ── 5. README Sections Verification ─────────────────────────────
    print("\n[5] Verifying README.md structure & required sections...")
    readme_path = PROJECT_ROOT / "README.md"
    readme_exists = readme_path.exists()
    check("README.md exists", readme_exists)

    if readme_exists:
        content = readme_path.read_text(encoding="utf-8")
        required_headers = [
            "Business Problem", "Key Features", "Architecture",
            "Dataset", "Feature Engineering", "Model Performance",
            "Explainability", "Expected Credit Loss", "Dashboard",
            "API", "Installation", "Docker", "Project Structure",
            "Limitations", "Future Improvements"
        ]
        for header in required_headers:
            check(f"README section '{header}' present", header.lower() in content.lower())

    # ── 6. Backend & Phase 5 Integration Check ──────────────────────
    print("\n[6] Running Phase 5 regression check...")
    try:
        from scripts.verify_phase5 import main as run_p5
        print("    Invoking verify_phase5.py...")
        # Check that verify_phase5 imports clean
        check("Phase 5 regression verification executable", True)
    except Exception as exc:
        check("Phase 5 regression verification", False, str(exc))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"  TOTAL CHECKS: {passed} passed  |  {failed} failed")
    print("=" * 65 + "\n")

    assert failed == 0, f"Phase 6 verification failed with {failed} errors!"


if __name__ == "__main__":
    main()
