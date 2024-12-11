# agents.py
"""
Agent implementations for the market research system.
Includes base agent class and specialized agents for different aspects of market research.
"""
import os
from datetime import datetime
import json
import time
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from research_agent.utils import AgentState, AgentType, MODEL_NAME, TEMPERATURE, AgentStatus
from research_agent.prompts import (
    BASE_PROMPT, MARKET_TRENDS_ROLE, COMPETITOR_ROLE,
    CONSUMER_ROLE, REPORT_ROLE
)

from langchain_community.tools.tavily_search import TavilySearchResults
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any, Optional, Callable
from pydantic import BaseModel

# Global tools setup
search_tool = TavilySearchResults(max_results=4)

# Model definition
model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

class MarketResearchState(TypedDict):
    """State for the market research workflow"""
    messages: List[AnyMessage]
    research_data: dict
    next_agent: str
    final_report: str | None
    _status_callback: Optional[Callable]
    focus_areas: List[str]

class SearchQueries(BaseModel):
    """Model for structured search queries"""
    queries: List[str]

def market_trends_node(state: MarketResearchState):
    """Node for market trends research"""
    focus_areas = state.get("focus_areas", [])
    print(f"[DEBUG] Market Trends Node - Focus Areas: {focus_areas}")

    # Skip if not in focus areas
    if "market_trends" not in focus_areas:
        print("[DEBUG] Skipping Market Trends Node")
        return {
            **state,
            "next_agent": "competitor",
        }

    status_callback = state.get("_status_callback")
    if status_callback:
        status_callback(AgentStatus.MARKET_TRENDS_START)
    start_time = time.time()

    queries = model.with_structured_output(SearchQueries).invoke([
        SystemMessage(content=MARKET_TRENDS_ROLE),
        HumanMessage(content=state['messages'][-1].content if state['messages'] else "Analyze market trends")
    ])

    research_data = state.get('research_data', {})

    # Perform searches and collect results
    search_results = []
    for query in queries.queries:
        results = search_tool.invoke({"query": query})
        search_results.extend(results)

    # Process results with LLM
    response = model.invoke([
        SystemMessage(content=MARKET_TRENDS_ROLE),
        HumanMessage(content=f"Analyze these market trends:\n\n{json.dumps(search_results)}")
    ])

    research_data['market_trends'] = {
        "last_update": datetime.now().isoformat(),
        "findings": response.content,
        "search_results": search_results
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    if status_callback:
        status_callback(f"{AgentStatus.MARKET_TRENDS_COMPLETE} (took {elapsed_time:.2f} seconds)")

    return {
        "messages": state.get('messages', []) + [response],
        "research_data": research_data,
        "next_agent": "competitor",
        "final_report": state.get("final_report", ""),
        "_status_callback": status_callback,
        "focus_areas": focus_areas
    }

def competitor_node(state: MarketResearchState):
    """Node for competitor analysis"""
    focus_areas = state.get("focus_areas", [])
    print(f"[DEBUG] Competitor Node - Focus Areas: {focus_areas}")

    # Skip if not in focus areas
    if "competitor_analysis" not in focus_areas:
        print("[DEBUG] Skipping Competitor Node")
        return {
            **state,
            "next_agent": "consumer",
        }

    status_callback = state.get("_status_callback")
    if status_callback:
        status_callback(AgentStatus.COMPETITOR_START)
    start_time = time.time()

    queries = model.with_structured_output(SearchQueries).invoke([
        SystemMessage(content=COMPETITOR_ROLE),
        HumanMessage(content=state['messages'][-1].content if state['messages'] else "Analyze competitors")
    ])

    research_data = state.get('research_data', {})

    # Perform searches and collect results
    search_results = []
    for query in queries.queries:
        results = search_tool.invoke({"query": query})
        search_results.extend(results)

    # Process results with LLM
    response = model.invoke([
        SystemMessage(content=COMPETITOR_ROLE),
        HumanMessage(content=f"Analyze these competitor insights:\n\n{json.dumps(search_results)}")
    ])

    research_data['competitor'] = {
        "last_update": datetime.now().isoformat(),
        "findings": response.content,
        "search_results": search_results
    }

    end_time = time.time()
    elapsed_time = end_time - start_time
    if status_callback:
        status_callback(f"{AgentStatus.COMPETITOR_COMPLETE} (took {elapsed_time:.2f} seconds)")

    return {
        "messages": state.get('messages', []) + [response],
        "research_data": research_data,
        "next_agent": "consumer",
        "final_report": state.get("final_report", ""),
        "_status_callback": status_callback,
        "focus_areas": focus_areas
    }

def consumer_node(state: MarketResearchState):
    """Node for consumer analysis"""
    focus_areas = state.get("focus_areas", [])
    print(f"[DEBUG] Consumer Node - Focus Areas: {focus_areas}")

    # Skip if not in focus areas
    if "consumer_behavior" not in focus_areas:
        print("[DEBUG] Skipping Consumer Node")
        return {
            **state,
            "next_agent": "report",
        }

    status_callback = state.get("_status_callback")
    if status_callback:
        status_callback(AgentStatus.CONSUMER_START)
    start_time = time.time()

    queries = model.with_structured_output(SearchQueries).invoke([
        SystemMessage(content=CONSUMER_ROLE),
        HumanMessage(content=state['messages'][-1].content if state['messages'] else "Analyze consumer behavior")
    ])

    # Initialize research_data if it doesn't exist
    research_data = state.get('research_data', {})
    # print(f"[DEBUG] Consumer Node - Initial Research Data: {research_data}")

    # Perform searches and collect results
    search_results = []
    for query in queries.queries:
        results = search_tool.invoke({"query": query})
        search_results.extend(results)

    # Process results with LLM
    response = model.invoke([
        SystemMessage(content=CONSUMER_ROLE),
        HumanMessage(content=f"Analyze these consumer insights:\n\n{json.dumps(search_results)}")
    ])

    # Store the findings
    research_data['consumer'] = {
        "last_update": datetime.now().isoformat(),
        "findings": response.content,
        "search_results": search_results
    }

    # print(f"[DEBUG] Consumer Node - Updated Research Data: {research_data}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    if status_callback:
        status_callback(f"{AgentStatus.CONSUMER_COMPLETE} (took {elapsed_time:.2f} seconds)")

    updated_state = {
        "messages": state.get('messages', []) + [response],
        "research_data": research_data,
        "next_agent": "report",
        "final_report": state.get("final_report", ""),
        "_status_callback": status_callback,
        "focus_areas": focus_areas
    }
    print(f"[DEBUG] Consumer Node - Returning State Keys: {updated_state.keys()}")
    return updated_state

def report_node(state: MarketResearchState):
    """Node for generating final report"""
    print("[DEBUG] Report Node - Entering with State Keys:", state.keys())

    focus_areas = state.get("focus_areas", [])
    status_callback = state.get("_status_callback")
    if status_callback:
        status_callback(AgentStatus.REPORT_START)

    research_data = state.get('research_data', {})
    print("[DEBUG] Report Node - Research Data Keys:", research_data.keys())

    # Get the original query from the last message
    original_query = state['messages'][-1].content if state['messages'] else "No query provided"

    # Start with the query section
    report_content = f"""## Research Query
{original_query}

"""

    # Map focus areas to research data keys
    focus_area_mapping = {
        "market_trends": "market_trends",
        "competitor_analysis": "competitor",
        "consumer_behavior": "consumer"
    }

    # Check each focus area for data
    for focus_area in focus_areas:
        data_key = focus_area_mapping.get(focus_area)
        if data_key and data_key in research_data:
            findings = research_data[data_key].get('findings', '')
            if findings:
                print(f"[DEBUG] Report Node - Found {focus_area} data")
                report_content += f"## {focus_area.replace('_', ' ').title()}\n{findings}\n\n"

    # Generate report if we have content
    if report_content:
        print("[DEBUG] Report Node - Generating report")
        report_prompt = f"Based on our research:\n\n{report_content}\n\nPlease generate a comprehensive market research report that synthesizes these findings."

        response = model.invoke([
            SystemMessage(content=REPORT_ROLE),
            HumanMessage(content=report_prompt)
        ])

        final_report = f"{report_content}\n{response.content}"
        print(f"[DEBUG] Report Node - Generated Report Length: {len(final_report)}")

        if status_callback:
            status_callback(AgentStatus.REPORT_COMPLETE)

        return {
            **state,
            "final_report": final_report,
            "next_agent": END
        }
    else:
        print("[DEBUG] Report Node - No research sections found")
        error_msg = "No research data was found for the selected focus areas."
        if status_callback:
            status_callback(f"❌ Error: {error_msg}")
        raise RuntimeError(error_msg)

def should_continue(state: MarketResearchState):
    """Determine next node based on state"""
    current_agent = state["next_agent"]
    focus_areas = state.get("focus_areas", [])

    print(f"[DEBUG] Should Continue - Current Agent: {current_agent}, Focus Areas: {focus_areas}")

    # If we're at the END or report, stop
    if current_agent in [END, "report"]:
        return END

    # Map agents to their focus areas
    agent_to_focus = {
        "market_trends": "market_trends",
        "competitor": "competitor_analysis",
        "consumer": "consumer_behavior"
    }

    # If current agent is in focus areas, let it execute by returning its name
    if agent_to_focus.get(current_agent) in focus_areas:
        print(f"[DEBUG] Should Continue - Executing {current_agent}")
        return current_agent

    # If current agent isn't in focus areas, find next valid agent
    agent_sequence = ["market_trends", "competitor", "consumer"]
    try:
        current_idx = agent_sequence.index(current_agent)
        remaining_agents = agent_sequence[current_idx + 1:]
    except ValueError:
        remaining_agents = []

    # Look for the next agent that matches a selected focus area
    for next_agent in remaining_agents:
        if agent_to_focus[next_agent] in focus_areas:
            print(f"[DEBUG] Should Continue - Moving to {next_agent}")
            return next_agent

    # If no more matching agents, go to report
    print("[DEBUG] Should Continue - Moving to report")
    return "report"

def build_research_graph():
    """Build the research workflow graph"""
    builder = StateGraph(MarketResearchState)

    # Add nodes
    builder.add_node("market_trends", market_trends_node)
    builder.add_node("competitor", competitor_node)
    builder.add_node("consumer", consumer_node)
    builder.add_node("report", report_node)

    # Set entry point
    builder.set_entry_point("market_trends")

    # Add conditional edges
    builder.add_conditional_edges(
        "market_trends",
        should_continue,
        {
            AgentType.COMPETITOR.value: "competitor",
            END: END
        }
    )

    builder.add_conditional_edges(
        "competitor",
        should_continue,
        {
            AgentType.CONSUMER.value: "consumer",
            END: END
        }
    )

    builder.add_conditional_edges(
        "consumer",
        should_continue,
        {
            AgentType.REPORT.value: "report",
            END: END
        }
    )

    builder.add_conditional_edges(
        "report",
        should_continue,
        {END: END}
    )

    return builder.compile()
