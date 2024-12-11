import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Union
from enum import Enum
import operator
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage
from markdown_pdf import MarkdownPdf, Section
import re
from datetime import datetime

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    research_data: dict
    final_report: Union[str, None]

class AgentType(str, Enum):
    MARKET_TRENDS = "market_trends"
    COMPETITOR = "competitor"
    CONSUMER = "consumer"
    REPORT = "report"
    END = "end"


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0

class AgentStatus:
    """Standardized status messages for agent workflow"""
    MARKET_TRENDS_START = "🔍 Starting Market Trends Analysis..."
    MARKET_TRENDS_COMPLETE = "✅ Market Trends Analysis complete"
    COMPETITOR_START = "🏢 Starting Competitor Analysis..."
    COMPETITOR_COMPLETE = "✅ Competitor Analysis complete"
    CONSUMER_START = "👥 Starting Consumer Behavior Analysis..."
    CONSUMER_COMPLETE = "✅ Consumer Analysis complete"
    REPORT_START = "📝 Starting Final Report Generation..."
    REPORT_COMPLETE = "✅ Final Report Generation complete"
    WAITING = "⏳ Waiting to start..."

# Progress mapping for UI updates
PROGRESS_MAP = {
    AgentStatus.MARKET_TRENDS_START: 0.05,
    AgentStatus.MARKET_TRENDS_COMPLETE: 0.39,
    AgentStatus.COMPETITOR_START: 0.4,
    AgentStatus.COMPETITOR_COMPLETE: 0.59,
    AgentStatus.CONSUMER_START: 0.6,
    AgentStatus.CONSUMER_COMPLETE: 0.79,
    AgentStatus.REPORT_START: 0.8,
    AgentStatus.REPORT_COMPLETE: 0.9,
}

PDF_CSS = """
    body { font-family: Arial, sans-serif; }
    h1 { color: #2c3e50; margin-top: 2em; }
    h2 { color: #34495e; margin-top: 1.5em; }
    h3 { color: #455a64; margin-top: 1.2em; }
    table, th, td { border: 1px solid #ddd; padding: 8px; }
    table { border-collapse: collapse; width: 100%; }
    th { background-color: #f5f5f5; }
    .toc { margin: 2em 0; }
    .toc ul { list-style-type: none; }
    .toc li { margin: 0.5em 0; }
"""


def create_pdf_from_markdown(markdown_content: str, output_file: str, title: str = "Market Research Report") -> bool:
    """
    Convert markdown content to PDF with proper sections and table of contents.
    """

    # Initialize PDF with table of contents up to level 3
    pdf = MarkdownPdf(toc_level=3)

    # Set PDF metadata
    pdf.meta["title"] = title
    pdf.meta["author"] = "Market Research Assistant"

    # Extract query if it exists (assuming it's in ## Research Query section)
    query_match = re.search(r'## Research Query\n(.*?)\n\n', markdown_content, re.DOTALL)
    query = query_match.group(1) if query_match else "No query provided"

    print(f"[DEBUG] Extracted query: {query}")  # Debug print

    # Add title page with query
    title_section = f"""# {title}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Research Query
{query}

"""
    # Add title page with TOC enabled
    pdf.add_section(Section(title_section, toc=3), user_css=PDF_CSS)

    # Split content into sections based on h1 headers (# )
    sections = re.split(r'(?=^# )', markdown_content.strip(), flags=re.MULTILINE)

    # Process sections
    for section in sections:
        if section.strip():  # Skip empty sections
            # Skip the original query section and title sections
            if not section.startswith('## Research Query') and not section.startswith('# Market Research Report\n\nGenerated on:'):
                pdf.add_section(Section(section), user_css=PDF_CSS)

    # Save the PDF
    pdf.save(output_file)
    return True
