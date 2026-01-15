import os
import shutil
from typing import Annotated, Literal, TypedDict
from pathlib import Path
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Loaders & Vector Stack
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --- CONFIGURATION ---
VECTOR_DB_PATH = "./sorterra_memory"
TEST_FOLDER = "./test_folder"
# Using a lightweight local embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_RECIPE = {
    "name": "Intelligent Project Sort",
    "rules": [
        "1. If a file is an invoice, move it to 'Finance/Invoices'.",
        "2. For other documents, find the project name (e.g., 'Project Alpha', 'Project Beta').",
        "3. Move those files to 'Projects/[Project Name]'.",
        "4. If no project is found, move it to 'Unsorted'."
    ]
}

# --- 1. VECTOR MEMORY UTILS ---

class SorterraMemory:
    def __init__(self):
        self.db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=EMBEDDING_MODEL)

    def get_similar_mapping(self, content: str):
        """Finds where similar files were previously moved."""
        try:
            results = self.db.similarity_search_with_relevance_scores(content, k=2)
            if not results or results[0][1] < 0.4: # Similarity threshold
                return "No high-confidence matches in memory."
            
            hints = []
            for doc, score in results:
                category = doc.metadata.get("destination", "unknown")
                hints.append(f"Previously sorted to '{category}' (Confidence: {score:.2f})")
            return "\n".join(hints)
        except Exception:
            return "Memory is currently empty."

    def learn_new_move(self, content: str, destination: str):
        """Indexes a successfully moved file for future reference."""
        self.db.add_texts(
            texts=[content],
            metadatas=[{"destination": destination}]
        )
        print(f"DEBUG: Sorterra learned that this content belongs in {destination}")

memory = SorterraMemory()

# --- 2. TOOLS ---

@tool
def read_file_content(file_path: str):
    """Reads PDF, DOCX, or TXT content. Essential for identifying project context."""
    path = Path(file_path)
    if not path.exists(): return f"Error: {file_path} not found."
    try:
        if path.suffix == ".pdf": loader = PyPDFLoader(str(path))
        elif path.suffix == ".docx": loader = Docx2txtLoader(str(path))
        else: loader = TextLoader(str(path))
        
        docs = loader.load()
        full_text = " ".join([d.page_content for d in docs])
        return full_text[:2000] # Cap for context window
    except Exception as e:
        return f"Error reading {path.name}: {str(e)}"

@tool
def get_past_examples(file_path: str):
    """Checks the vector database for where similar files were sorted in the past."""
    content = read_file_content.invoke(file_path)
    if "Error" in content: return content
    return memory.get_similar_mapping(content)

@tool
def move_file(source_path: str, destination_folder: str):
    """Moves file and triggers the 'learning' process for the vector DB."""
    source = Path(source_path)
    dest_dir = Path(destination_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Before moving, read content so we can 'learn' it
        content = read_file_content.invoke(source_path)
        
        # Perform move
        shutil.move(str(source), str(dest_dir / source.name))
        
        # Update Vector Memory
        if "Error" not in content:
            memory.learn_new_move(content, destination_folder)
            
        return f"Successfully moved {source.name} to {destination_folder} and updated memory."
    except Exception as e:
        return f"Failed to move: {str(e)}"

@tool
def list_local_files(directory: str):
    """Lists files to be sorted."""
    path = Path(directory)
    if not path.is_dir(): return f"Error: {directory} is not a directory."
    return [str(f) for f in path.iterdir() if f.is_file()]

tools = [move_file, list_local_files, read_file_content, get_past_examples]
tool_node = ToolNode(tools)

# --- 3. GRAPH LOGIC ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "Conversation history"]
    recipe: dict

def get_model(local=False):
    if local:
        return ChatOllama(model="deepseek-r1:8b", temperature=0).bind_tools(tools)
    return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0).bind_tools(tools)

def call_model(state: AgentState):
    recipe = state.get("recipe", DEFAULT_RECIPE)
    rules_str = "\n".join(recipe["rules"])
    
    system_prompt = SystemMessage(content=(
        f"You are the Sorterra Agent. Your goal: Sort files using '{recipe['name']}'.\n\n"
        f"CORE RULES:\n{rules_str}\n\n"
        "HYBRID STRATEGY:\n"
        "1. List files in the target directory.\n"
        "2. For EACH file, use 'get_past_examples' to see vector similarity hints.\n"
        "3. Use 'read_file_content' to confirm against the Core Rules.\n"
        "4. If Vector Hints and Core Rules align, move the file.\n"
        "5. If rules are ambiguous (e.g. 'Project Alpha' isn't explicitly named), "
        "trust the high-confidence vector matches."
    ))
    
    response = model.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state['messages'][-1]
    return "tools" if last_message.tool_calls else "__end__"

# Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()

if __name__ == "__main__":
    model = get_model(local=False) # Toggle to True for your Ollama setup
    
    inputs = {
        "messages": [HumanMessage(content=f"Sort the files in {TEST_FOLDER}")],
        "recipe": DEFAULT_RECIPE
    }
    
    for output in app.stream(inputs, stream_mode="updates"):
        for node, values in output.items():
            if "messages" in values:
                last_msg = values["messages"][-1]
                if last_msg.content:
                    print(f"[{node}]: {last_msg.content}")