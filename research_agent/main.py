# main.py
"""
Main execution script for the market research system.
Provides interface for running market research analysis.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from research_agent.workflow import create_market_research_orchestrator

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Market Research Agent CLI")
    parser.add_argument(
        "--reports-dir",
        type=str,
        help="Directory to store reports (default: repository's reports folder)",
        default=None
    )
    return parser.parse_args()

def print_status(message: str):
    """Print a status update with emoji indicators"""
    print(f"\n{message}")

def run_research(query: str, reports_dir: Optional[str] = None) -> Optional[dict]:
    """
    Run the market research workflow

    Args:
        query: Research query to analyze
        reports_dir: Optional directory to store reports

    Returns:
        Optional[dict]: Research results if successful, None if failed
    """
    try:
        storage_config = {"base_dir": reports_dir} if reports_dir else {}

        # Initialize orchestrator with status callback
        orchestrator = create_market_research_orchestrator(
            storage_type="local",
            storage_config=storage_config,
            status_callback=print_status
        )

        # Run research
        return orchestrator.run_research(query)

    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        return None

def print_results(results: dict):
    """
    Print research results in a formatted way

    Args:
        results: Dictionary containing research results
    """
    print("\n📊 Research Results")
    print("=" * 80)
    print(results["final_report"])
    print("=" * 80)

    print("\n📁 Generated Files:")
    print(f"- Report: {results['report_info']['path']}")
    if results.get('findings_info'):
        print(f"- Findings: {results['findings_info']['path']}")

def main():
    """Main entry point for the market research CLI"""
    try:
        args = parse_args()

        # Get user input
#         query = """Conduct a comprehensive market analysis of wearable fitness trackers,
# focusing on current trends, major competitors, and consumer preferences.
# Pay special attention to emerging technologies and integration opportunities
# with personalized wellness coaching systems."""
#         print("Using query:")
#         print('---')
#         print(query)
#         print('---')
        query = input("\n🔍 Enter your market research query: ").strip()
        if not query:
            print("\n❌ Query cannot be empty")
            return 1

        print("\n🚀 Starting market research process...")

        # Run research with optional custom reports directory
        results = run_research(query, args.reports_dir)
        if not results:
            return 1

        # Print results
        print_results(results)
        print("\n✨ Market research complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Research interrupted by user")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
