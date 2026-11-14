"""
Main Entry Point for Ashcombe & Rowe Estates Chatbot Application.
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Ashcombe & Rowe Estates AI Lead Qualification System")
    parser.add_argument(
        "--mode",
        choices=["cli", "api", "evaluate"],
        default="cli",
        help="Run mode: 'cli' (terminal chat), 'api' (FastAPI server), or 'evaluate' (benchmark suite).",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for FastAPI server")

    args = parser.parse_args()

    if args.mode == "cli":
        from cli import run_cli
        run_cli()
    elif args.mode == "api":
        import uvicorn
        print(f"Starting Ashcombe & Rowe Estates FastAPI server on http://{args.host}:{args.port}")
        uvicorn.run("api:app", host=args.host, port=args.port, reload=True)
    elif args.mode == "evaluate":
        from evaluate import run_evaluations
        run_evaluations()


if __name__ == "__main__":
    main()
