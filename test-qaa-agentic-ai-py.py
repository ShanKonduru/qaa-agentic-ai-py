import os
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

# Load environment variables (if any, though for Ollama, it's mostly config_list)
load_dotenv()

# --- Configuration for Ollama LLM ---
# This loads from OAI_CONFIG_LIST file.
# Make sure your OAI_CONFIG_LIST file is correctly configured for Ollama.
llm_config = {
    "config_list": autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["ollama/mistral"], # Ensure this matches the model name in your OAI_CONFIG_LIST
        },
    ),
    "temperature": 0.7, # Adjust for creativity (higher) vs. predictability (lower)
}

# --- Define Agents ---

# User Proxy Agent: Interacts with the user and executes code
# For code execution, we'll use a local command line executor.
# Consider using DockerCommandLineCodeExecutor for sandboxed execution in production.
code_execution_work_dir = "coding_workspace"
os.makedirs(code_execution_work_dir, exist_ok=True) # Ensure the directory exists

user_proxy = UserProxyAgent(
    name="Admin",
    system_message="A human administrator who oversees the process, provides tasks, and reviews outputs. Can execute code.",
    llm_config=llm_config, # It can also use the LLM for chat if needed
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(work_dir=code_execution_work_dir),
        "last_n_messages": 1, # Only look at the last message for code execution
    },
    human_input_mode="ALWAYS", # Always ask for human input before executing code or terminating
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Test Plan Generator Agent
test_plan_generator = AssistantAgent(
    name="TestPlanGenerator",
    system_message="You are an expert Test Lead. Your role is to understand the user's request and create a detailed test plan, outlining the high-level scenarios, types of testing needed (functional, negative, boundary, performance), and the scope. Output the plan in a clear, structured format.",
    llm_config=llm_config,
)

# Test Case Generator Agent
test_case_generator = AssistantAgent(
    name="TestCaseGenerator",
    system_message="You are a meticulous Test Case Designer. Based on the test plan, you will generate detailed, step-by-step test cases. For each test case, include a Test ID, Description, Preconditions, Steps, and Expected Result. Use a clear, tabular, or list format.",
    llm_config=llm_config,
)

# Code Generator Agent
code_generator = AssistantAgent(
    name="CodeGenerator",
    system_message="You are a skilled Test Automation Engineer. Your task is to write Python automation scripts (e.g., using Playwright for UI or 'requests' for API) based on the provided test cases. Ensure the code is robust, includes assertions, and handles common errors. Provide only the code, enclosed in triple backticks. If you need to install a library, suggest `pip install <library_name>`.",
    llm_config=llm_config,
)

# Test Report Analyzer Agent
test_report_analyzer = AssistantAgent(
    name="TestReportAnalyzer",
    system_message="You are an insightful QA Analyst. Analyze the test execution logs and results. Summarize the findings, identify any failures or anomalies, provide root cause analysis if possible, and suggest actionable recommendations for bug fixing or test suite improvement. Format your report clearly.",
    llm_config=llm_config,
)

# --- Define the Workflow (Group Chat) ---

# Create a group chat for the agents to collaborate
groupchat = autogen.GroupChat(
    agents=[user_proxy, test_plan_generator, test_case_generator, code_generator, test_report_analyzer],
    messages=[],
    max_round=15, # Limit conversation rounds to prevent infinite loops
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Start the Conversation ---

if __name__ == "__main__":
    initial_task = input("What do you want to test (e.g., 'the login functionality of a web application with valid and invalid credentials')? \n")

    user_proxy.initiate_chat(
        manager,
        message=f"Please create a detailed test plan, then generate test cases, write automation code, execute it, and finally analyze the results for: {initial_task}",
    )

    print(f"\n--- Conversation Finished ---")
    print(f"Check the '{code_execution_work_dir}' directory for any generated code or files.")