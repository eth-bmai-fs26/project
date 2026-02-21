 # imports
import os
from openai import OpenAI


#make environemnt var
token = "add your token here"

def llm_chat(user_query):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    ) 

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
        messages=[
            {
                "role": "user",
                "content": user_query,

            }

        ],
    )
    
    return completion


import pandas as pd
from openai import OpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import json

# Load the dataset once at module level
file_path = "Data/fashion_data_2018_2022.xls"
df = pd.read_excel(file_path)

@tool
def get_fashion_inspiration(
    category: str = "",
    year: int = None,
    season: str = ""
) -> str:
    """
    Retrieves fashion inspiration and trends from historical fashion data (2018–2022).
    Use this tool to generate fashion proposals grounded in real trend data.
    """
    filtered = df.copy()

    
    
    return filtered


def build_tool_prompt(user_query: str, tool_result: str) -> str:
    """Builds a prompt that incorporates tool output into the LLM request."""
    return f"""You are a fashion expert. Using the following real trend data as inspiration, generate a creative and detailed fashion proposal for the user.

TREND DATA:
{tool_result}

USER REQUEST:
{user_query}

Generate a thoughtful fashion proposal inspired by the data above."""


def llm_chat_with_tools(user_query: str, **tool_kwargs):
    """
    Enhanced llm_chat that uses the fashion inspiration tool before calling the LLM.
    
    Args:
        user_query: The user's fashion request.
        **tool_kwargs: Optional filters passed to get_fashion_inspiration 
                       (category, year, season).
    """
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    # Step 1: Invoke the tool
    tool_result = get_fashion_inspiration.invoke(tool_kwargs)

    # Step 2: Build enriched prompt
    enriched_prompt = build_tool_prompt(user_query, tool_result)

    # Step 3: Call the LLM with the enriched context
    completion = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507:novita",
        messages=[
            {
                "role": "user",
                "content": enriched_prompt,
            }
        ],
    )

    return completion
