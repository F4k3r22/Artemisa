from .config import SearchEngine, Configuration
from .prompts import *
from .state import *
from .utils import *
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from Artemisa.Llm.local import OllamaLocal
from typing_extensions import Literal
import json

def GenerateQuery(state: SummaryState, config: RunnableConfig):

    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = OllamaLocal(model=configurable.local_llm, format="json")
    result = llm_json_mode.query(
        f"Generate a query for web search:",
        query_writer_instructions_formatted
        
    )
    query = json.loads(result.content)

    return {"search_query": query['query']}

def WebResearch(state: SummaryState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)

    search_results = SearchEngine(state.search_query, num_search=3, fetch_full_page=configurable.fetch_full_page)
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def SummarizeSources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = OllamaLocal(model=configurable.local_llm,)
    result = llm.query(
        human_message_content,
        summarizer_instructions
        
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def ReflectOnSummary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = OllamaLocal(model=configurable.local_llm, format="json")
    result = llm_json_mode.query(
        f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}",
        reflection_instructions.format(research_topic=state.research_topic)
    )

    follow_up_query = json.loads(result.content)

    query = follow_up_query.get('follow_up_query')

    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

def FinalizeSummary(state: SummaryState):
    """ Finalize the summary """
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["FinalizeSummary", "WebResearch"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "WebResearch"
    else:
        return "FinalizeSummary"
    

builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("GenerateQuery", GenerateQuery)
builder.add_node("WebResearch", WebResearch)
builder.add_node("SummarizeSources", SummarizeSources)
builder.add_node("ReflectOnSummary", ReflectOnSummary)
builder.add_node("FinalizeSummary", FinalizeSummary)

builder.add_edge(START, "GenerateQuery")
builder.add_edge("GenerateQuery", "WebResearch")
builder.add_edge("WebResearch", "SummarizeSources")
builder.add_edge("SummarizeSources", "ReflectOnSummary")
builder.add_conditional_edges("ReflectOnSummary", route_research)
builder.add_edge("FinalizeSummary", END)

DeepResearcher = builder.compile()