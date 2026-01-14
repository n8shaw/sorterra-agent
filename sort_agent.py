import os
import shutil
from typing import Annotated, Literal, TypedDict
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- 1. DYNAMIC RECIPE CONFIGURATION ---
# In the future, this dictionary could be fetched from a JSON file or DB
DEFAULT_RECIPE = {
    "name": "Standard Sorting",
    "rules": [
        ".pdf and .docx files go to 'Documents'",
        ".jpg and .png files go to 'Images'",
        "Any file with 'invoice' in the name goes to 'Finance'"
    ]
}

def get_sorting_instructions(recipe: dict) -> str:
    """Formats the recipe dictionary into a system prompt string."""
    rules_str = "\n".join([f"- {rule}" for rule in recipe["rules"]])
    return (
        f"You are the Sorterra Agent. Your job is to sort files based on the '{recipe['name']}' recipe:\n"
        f"{rules_str}\n\n"
        "Instructions:\n"
        "1. First, list the files in the directory to see what needs sorting.\n"
        "2. Analyze the filenames and extensions.\n"
        "3. Move the files one by one to the appropriate folders according to the recipe rules.\n"
        "4. If a file doesn't match any rule, leave it where it is."
    )

# --- 2. TOOLS (CLOUD-PORTABLE LOGIC) ---
@tool
def move_file(source_path: str, destination_folder: str):
    """Moves a file to a specific destination folder. Use this for sorting."""
    source = Path(source_path)
    dest_dir = Path(destination_folder)
    
    if not source.exists():
        return f"Error: File {source_path} does not exist."
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(source), str(dest_dir / source.name))
        return f"Successfully moved {source.name} to {destination_folder}"
    except Exception as e:
        return f"Failed to move file: {str(e)}"

@tool
def list_local_files(directory: str):
    """Lists all files in a directory to identify what needs sorting."""
    path = Path(directory)
    if not path.is_dir():
        return f"Error: {directory} is not a valid directory."
    
    files = [f.name for f in path.iterdir() if f.is_file()]
    return f"Files found in {directory}: {', '.join(files)}"

tools = [move_file, list_local_files]
tool_node = ToolNode(tools)

# --- 3. GRAPH STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "The conversation history"]
    # We can add 'recipe' to the state here if we want to pass it dynamically per-run
    recipe: dict 

# --- 4. MODEL SELECTION ---
def get_model(local=False):
    if local:
        return ChatOllama(model="deepseek-r1:8b", temperature=0).bind_tools(tools)
    else:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0).bind_tools(tools)

# --- 5. AGENT NODE ---
def call_model(state: AgentState):
    messages = state['messages']
    # Fetch the recipe from state (or default if not provided)
    recipe = state.get("recipe", DEFAULT_RECIPE)
    
    system_prompt = SystemMessage(content=get_sorting_instructions(recipe))
    
    response = model.invoke([system_prompt] + messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    return "tools" if last_message.tool_calls else END

# --- 6. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- 7. EXECUTION ---
if __name__ == "__main__":
    USE_LOCAL = False 
    model = get_model(local=USE_LOCAL)

    print(f"--- Sorterra Running ({'Local' if USE_LOCAL else 'Cloud'}) ---")
    
    # You can now pass a CUSTOM recipe here if you want to override the default
    inputs = {
        "messages": [HumanMessage(content="Sort the files in ./test_folder")],
        "recipe": DEFAULT_RECIPE 
    }
    
    for output in app.stream(inputs, stream_mode="updates"):
        for node, values in output.items():
            if "messages" in values:
                last_msg = values["messages"][-1]
                # Log model's reasoning/action
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(f"[{node}]: {last_msg.content}")
                if last_msg.tool_calls:
                    print(f"[{node}]: Calling tools: {[t['name'] for t in last_msg.tool_calls]}")