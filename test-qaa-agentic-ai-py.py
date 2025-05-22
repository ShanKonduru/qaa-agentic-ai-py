import os
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

import tempfile

# Load environment variables (if any, though for Ollama, it's mostly config_list)
load_dotenv()

# --- Configuration for Ollama LLM ---
llm_config = {
    "config_list": [
        {
            "model": "llama2:13b",
            "api_type": "ollama",
            "base_url": "http://localhost:11434/api"
        },
        {
            "model": "deepseek-r1:8b",
            "api_type": "ollama",
            "base_url": "http://localhost:11434/api"
        }
    ],
    "temperature": 0.7,
    "timeout": 600
}


def init_log():
    # Define the log directory
    log_directory = "logs"
    # This creates "logs/conversation_log.log"
    log_filename = os.path.join(log_directory, "conversation_log.log")

    # Create the directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        print(f"Created directory: {log_directory}")

    # Start Autogen runtime logging
    logging_session_id = autogen.runtime_logging.start(
        logger_type="file",
        config={"filename": log_filename}
    )

    print(f"AutoGen logging started. Session ID: {logging_session_id}")
    print(f"Logs will be saved to '{log_filename}'.")
    return logging_session_id


# --- Define Agents ---
# User Proxy Agent: Interacts with the user and executes code
# For code execution, we'll use a local command line executor.
# Consider using DockerCommandLineCodeExecutor for sandboxed execution in production.
code_execution_work_dir = "coding_workspace"
# Ensure the directory exists
os.makedirs(code_execution_work_dir, exist_ok=True)

# --- Start AutoGen Runtime Logging to a .log file ---
# Create a temporary directory for logs if you want them isolated, otherwise use current dir.
# Use this if you want logs in a temp dir
current_log_dir = tempfile.TemporaryDirectory()
logging_session_id = init_log()

user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human administrator who oversees the process, provides tasks, and reviews outputs. Can execute code.",
    llm_config=llm_config,
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir=code_execution_work_dir),
        "last_n_messages": 1,
    },
    # Always ask for human input before executing code or terminating
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
)

# Test Plan Generator Agent
test_plan_generator = AssistantAgent(
    name="TestPlanGenerator",
    system_message="""You are an expert Test Lead.
Your role is to understand the user's request and create a detailed test plan.
Output the plan in a clear, structured format, typically including the following sections:

- **Introduction/Objective:** Briefly state the purpose of the test plan.
- **Test Scope:** Clearly define what will be tested (in-scope) and what will not be tested (out-of-scope).
- **Types of Testing:** Outline the various testing methodologies to be employed (e.g., functional, negative, boundary, performance, security, usability, regression, integration).
- **High-Level Test Scenarios:** Describe the main functionalities or user flows that will be tested. Focus on *what* to test, not detailed steps.
- **Test Environment Requirements:** Specify any necessary hardware, software, or network configurations.
- **Test Data Requirements:** Briefly outline the types of test data needed.
- **Entry Criteria:** Define the conditions that must be met before testing can begin.
- **Exit Criteria:** Define the conditions that must be met for testing to be considered complete.
- **Risks & Mitigations:** Identify potential risks to the testing effort and plans to mitigate them.

Explain the rationale behind the proposed testing approaches and scope decisions.
Once the test plan is complete, ask the user for approval or any necessary modifications.
""",
    llm_config=llm_config,
)

# Test Case Generator Agent
test_case_generator = AssistantAgent(
    name="TestCaseGenerator",
    system_message="""
You are a meticulous Test Case Designer. Based on the test plan, you will generate detailed, step-by-step test cases.
For each test case, include a **Test ID**, **Description**, **Preconditions**, **Steps** (as actionable instructions), **Expected Result**, and a placeholder for **Actual Result**.
Clearly indicate the **Test Case Type** (e.g., Functional, Negative, Boundary) where applicable.
Use a clear, tabular, or list format for presentation.    
    """,
    llm_config=llm_config,
)

# Code Generator Agent
code_generator = AssistantAgent(
    name="CodeGenerator",
    system_message="""
You are a skilled Test Automation Engineer.
Your task is to write complete and runnable Python automation scripts (e.g., using Playwright for UI or 'requests' for API) based on the provided test cases.
Ensure the code is robust, includes clear assertions, handles common errors gracefully, and is designed for reusability and maintainability.
For UI automation, prioritize reliable locators and appropriate waiting strategies.
Provide only the code, enclosed in triple backticks.
If you need to install a library, suggest `pip install <library_name>`.    
    """,
    llm_config=llm_config,
)

# Test Report Analyzer Agent
test_report_analyzer = AssistantAgent(
    name="TestReportAnalyzer",
    system_message="""
You are an insightful QA Analyst.
Your task is to thoroughly analyze test execution logs and results.
Summarize the findings, identify any failures or anomalies, and provide root cause analysis where possible.
Based on your analysis, suggest actionable recommendations for bug fixing or test suite improvement.
Format your report clearly, typically including sections such as 'Summary of Findings,' 'Failed Test Cases/Anomalies,' 'Root Cause Analysis,' and 'Recommendations for Improvement.'    
    """,
    llm_config=llm_config,
)

# --- Define the Workflow (Group Chat) ---

# Create a group chat for the agents to collaborate
groupchat = autogen.GroupChat(
    agents=[user_proxy, test_plan_generator, test_case_generator,
            code_generator, test_report_analyzer],
    messages=[],
    max_round=5,  # Limit conversation rounds to prevent infinite loops
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Start the Conversation ---

if __name__ == "__main__":
    initial_task = input(
        "What do you want to test (e.g., 'the login functionality of a web application with valid and invalid credentials')? \n")

    user_proxy.initiate_chat(
        manager,
        message=f"Please create a detailed test plan, then generate test cases, write automation code, execute it, and finally analyze the results for: {initial_task}",
    )

    print(f"\n--- Conversation Finished ---")
    print(
        f"Check the '{code_execution_work_dir}' directory for any generated code or files.")

    # --- Stop AutoGen Runtime Logging ---
    autogen.runtime_logging.stop()
    print("AutoGen logging stopped. You can now review 'conversation_log.log'.")
