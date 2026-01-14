# Sorterra: Local Sorting Agent (v1.0)

**Sorterra** is a "Cloud-Ready" file management agent built using **LangGraph 1.0**. It uses an agentic loop to analyze, categorize, and move files based on a configurable "Recipe." By decoupling the file logic from the agent's brain, this system is designed to transition from your local machine to SharePoint with minimal friction.

---

### Project Overview
The agent operates on a **Sense-Think-Act** loop:
1. **Sense:** The agent uses the `list_local_files` tool to see what is in your target folder.
2. **Think:** It compares the filenames and extensions against the **Dynamic Recipe**.
3. **Act:** It uses the `move_file` tool to execute the sorting logic.

---

### File Structure
* `sorterra_agent.py`: The main Python script containing the graph and tools.
* `requirements.txt`: List of necessary Python libraries.
* `.env`: Your private API keys (not to be committed to version control).
* `test_folder/`: The local directory where the agent will perform its work.

---

### How to Run Locally

#### 1. Prerequisites
* **Python 3.11+**
* **Ollama** (For local mode): Download from [ollama.com](https://ollama.com).
* **Anthropic API Key** (For cloud mode): Get one from the [Anthropic Console](https://console.anthropic.com/).

#### 2. Environment Setup
Create a `.env` file in the root directory and add your key:
```bash
ANTHROPIC_API_KEY=your_key_here
```
#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
#### 4. Prepare Your Model (Local Only)
If you set `USE_LOCAL = True` in the code, pull a reasoning model via Ollama:

```bash
ollama pull deepseek-r1:8b
```
#### 5. Execution
Create a folder named `test_folder`.

Add dummy files (e.g., january_invoice.txt, family_photo.jpg, resume.pdf).

Run the agent:

```bash
python sorterra_agent.py
```