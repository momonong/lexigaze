"""Run the local reading-to-learner-profile development workspace."""

from __future__ import annotations

import argparse

from web.learner_workspace_app import create_learner_workspace_app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local LexiGaze reading and learner-profile workspace."
    )
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be an unprivileged port between 1024 and 65535")
    authority = f"127.0.0.1:{args.port}"
    app = create_learner_workspace_app({"LEXIGAZE_LEARNER_AUTHORITY": authority})
    print(f"Learner workspace: http://{authority}/learner-workspace")
    print("Local developer preview; session memory only; no camera or model inference.")
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
