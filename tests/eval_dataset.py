# tests/eval_dataset.py
EVAL_CASES = [
    {
        "input_file": "invoice_9999_test.txt",
        "content": "INVOICE #9999\nVendor: AWS\nTotal: $150.00\nDate: 2026-01-30\nDescription: Monthly cloud hosting.",
        "expected_folder": "Finance/Invoices",
        "expected_rename_required": True,
        "criteria": "The agent must identify the vendor as AWS and move it to the Finance/Invoices folder."
    },
    {
        "input_file": "alpha_plan_debug.txt",
        "content": "Project Alpha: Database Migration Phase. This doc outlines the SQL scripts for Alpha.",
        "expected_folder": "Projects/Alpha",
        "expected_rename_required": False,
        "criteria": "The agent must recognize 'Project Alpha' and sort it into the specific project subfolder."
    }
]