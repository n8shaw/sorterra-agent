# tests/evaluator.py
import json
import shutil
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Define the path first
VECTOR_DB_PATH = "./data/sorterra_memory"

def run_evals():
    """Iterates through test cases, runs the agent, and grades the results."""
    
    # 2. CLEAR MEMORY FIRST (Before importing the agent/tools)
    if Path(VECTOR_DB_PATH).exists():
        print(f"--- Clearing vector memory at {VECTOR_DB_PATH} ---")
        try:
            shutil.rmtree(VECTOR_DB_PATH)
        except PermissionError:
            print("⚠️ Warning: Could not clear memory. Ensure no other processes are using the DB.")

    # 3. IMPORT AGENT NOW (After the file lock is gone)
    from core.agent import app, model_thinking
    from tests.eval_dataset import EVAL_CASES

    results = []
    test_dir = Path("./data/test_folder")
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nSTARTING AGENTIC EVALUATIONS\n{'='*60}")
    
    for case in EVAL_CASES:
        # Setup: Create the physical file for the agent to analyze
        test_path = test_dir / case['input_file']
        test_path.write_text(case["content"])

        print(f"\n[TEST] File: {case['input_file']}")

        # Run the Sorterra Agent
        inputs = {
            "messages": [HumanMessage(content=f"Sort this file: {test_path}")],
            "recipe": {
                "name": "Project Sort",
                "rules": [
                    "1. If file is an invoice, move to 'Finance/Invoices'.",
                    "2. If it mentions a Project (e.g. Alpha), move to 'Projects/[Name]'.",
                    "3. Otherwise, 'Unsorted'."
                ]
            },
            "current_file": str(test_path)
        }

        try:
            # Execute graph and capture final state
            final_state = app.invoke(inputs)
            
            # Grading Logic (Pass model_thinking explicitly)
            report = grade_agent_action(case, final_state, model_thinking)
            results.append(report)
            
            status = "✅ PASS" if report['grade'] == 'PASS' else "❌ FAIL"
            print(f"RESULT: {status}")
            print(f"EXPLANATION: {report['judge_explanation']}")
        
        except Exception as e:
            print(f"ERROR processing {case['input_file']}: {str(e)}")

    # Summary Statistics
    print(f"\n{'='*60}")
    pass_count = sum(1 for r in results if r['grade'] == 'PASS')
    total = len(results)
    score = (pass_count / total) * 100 if total > 0 else 0
    print(f"FINAL SCORE: {pass_count}/{total} ({score:.1f}%)")
    print(f"{'='*60}\n")

def grade_agent_action(case, final_state, model_thinking):
    """Uses LLM-as-a-judge to evaluate the agent's performance."""
    
    all_tool_calls = []
    for msg in final_state["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            all_tool_calls.extend(msg.tool_calls)
    
    reasoning = final_state["messages"][-1].content

    system_prompt = "You are an impartial judge evaluating the performance of an AI file-management agent."
    
    judge_task = f"""
    Evaluate the following agent execution against the Target Criteria.

    ### TARGET CRITERIA
    - Required Folder: {case['expected_folder']}
    - Specific Goal: {case['criteria']}
    
    ### AGENT PERFORMANCE
    - Final Reasoning Provided: "{reasoning}"
    - Total Tool Calls Executed: {json.dumps(all_tool_calls, indent=2)}
    
    ### EVALUATION REQUIREMENTS
    1. GRADE: 'PASS' if ANY tool call correctly moved the file to '{case['expected_folder']}'. 
    2. EXPLANATION: Provide a concise 1-2 sentence justification.

    ### OUTPUT FORMAT
    You MUST return your response as a valid JSON object:
    {{
        "grade": "PASS" or "FAIL",
        "explanation": "concise explanation"
    }}
    """
    
    response = model_thinking.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=judge_task)
    ])
    
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].strip()

    try:
        judge_output = json.loads(content)
    except json.JSONDecodeError:
        return {
            "input": case['input_file'],
            "grade": "FAIL",
            "judge_explanation": f"Judge returned invalid JSON: {content}"
        }
    
    return {
        "input": case['input_file'],
        "grade": judge_output.get('grade', 'FAIL'),
        "judge_explanation": judge_output.get('explanation', 'No explanation provided.')
    }

if __name__ == "__main__":
    run_evals()