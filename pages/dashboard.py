import streamlit as st
from google.cloud import firestore
import pandas as pd
import os
from openai import OpenAI
from code_editor import code_editor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_openai import ChatOpenAI
import streamlit_shadcn_ui as ui
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import re
import streamlit_shadcn_ui as ui
import traceback
import json
from langchain.agents import AgentType, initialize_agent, load_tools


# Set the Google application credentials if not set
#if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
#    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'firebase/serviceAccountKey.json'

#if not os.environ.get("OPENAI_API_KEY"):
#    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

if "fetched_code" not in st.session_state:
    st.session_state["fetched_code"] = "Code not fetched"

if "events_and_descriptions" not in st.session_state:
    st.session_state["events_and_descriptions"] = "Events not fetched"

if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()

if "fetch_components" not in st.session_state:
    st.session_state["fetch_component"] = True

if "nuggts" not in st.session_state:
    st.session_state["nuggts"] = None

if "activation_disabled" not in st.session_state:
    st.session_state["activation_disabled"] = True

if "tracker_generated" not in st.session_state:
    st.session_state["tracker_generated"] = False

if "visuals_generated" not in st.session_state:
    st.session_state["visuals_generated"] = False

if "visuals_disabled" not in st.session_state:
    st.session_state["visuals_disabled"] = True

if "tracker" not in st.session_state:
    st.session_state["tracker"] = "#Python Code to track this metric will appear here"
    
if "visuals" not in st.session_state:
    st.session_state["visuals"] = "#Python Code for visualisations will appear here"

if "current_response" not in st.session_state:
    st.session_state["current_response"] = "Agent's response will appear here"

if "updated_code" not in st.session_state:
    st.session_state["updated_code"] = False

if "chosen_index" not in st.session_state:
    st.session_state["chosen_index"] = ""
if "chosen_heading" not in st.session_state:
    st.session_state["chosen_heading"] = ""
if "chosen_metric" not in st.session_state:
    st.session_state["chosen_metric"] = ""
if "chosen_formula" not in st.session_state:
    st.session_state["chosen_formula"] = ""
if "chosen_decision_tree" not in st.session_state:  
    st.session_state["chosen_decision_tree"] = ""
if "chosen_visualisatoins" not in st.session_state:
    st.session_state["chosen_visualisations"] = ""
if "chosen_tracker_code" not in st.session_state:
    st.session_state["chosen_tracker_code"] = ""
if "chosen_visuals_code" not in st.session_state:
    st.session_state["chosen_visuals_code"] = ""

if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Set the page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Firestore
db = firestore.Client()

introducing_session_id = """
Each session of user interaction is tracked with a unique session_id. A session represents a series of events 
that occur within a specific time frame, typically corresponding to a single visit or interaction by a user. For a single,
session_id, event data is captured in chronological order (based on timestamp).

Captured Events:\n
"""

# Function to fetch all component identifiers
def fetch_all_component_identifiers():
    components = db.collection('components').stream()
    component_identifiers = [component.id for component in components]
    return component_identifiers

# Function to fetch the code of a selected component
def fetch_component_code(component_identifier):
    component_ref = db.collection('components').document(component_identifier)
    component_data = component_ref.get().to_dict()
    return component_data.get('react_code', 'No code available')

# Function to fetch and group data by session
def fetch_and_group_data_by_session(component_identifier):
    # Reference to the component document
    component_ref = db.collection('components').document(component_identifier)
    
    # Create an empty list to store all event data
    all_events = []

    # Fetch all events under the component
    events = component_ref.collection('events').stream()

    for event in events:
        event_name = event.id  # Event name
        event_data = event.to_dict()  # Fetch event description
        event_description = event_data.get('event_description', 'No description available')

        event_data_ref = component_ref.collection('events').document(event_name).collection('data')
        event_instances = event_data_ref.stream()

        for event_instance in event_instances:
            event_instance_data = event_instance.to_dict()
            session_id = event_instance_data.get('session_id')

            # Ensure session_id is a string or convert it to one
            if isinstance(session_id, dict):
                session_id = list(session_id.values())[0]
            elif session_id is None:
                session_id = "unknown_session"

            # Handle 'value_data' and ensure proper handling of dictionary values
            value_data = {k: v for k, v in event_instance_data.items() if k not in ['session_id', 'timestamp']}
            
            # If value_data contains a dictionary, extract its first value
            if isinstance(value_data, dict) and value_data:
                value = list(value_data.values())[0]  # Get the first value from the dictionary
            elif value_data:
                value = list(value_data.values())[0] if isinstance(value_data, dict) else value_data
            else:
                value = event_name  # Default to event name if value_data is empty

            # Capture the data type of the value
            value_type = type(value).__name__

            # Append the event data
            all_events.append({
                'session_id': session_id,
                'event_name': event_name,
                'event_description': event_description,
                'value': value,
                'value_type': value_type,  # Add a new column for the data type
                'timestamp': event_instance_data.get('timestamp')
            })

    # Now that all events are collected, create a DataFrame from the list
    df = pd.DataFrame(all_events)

    # Group the data by 'session_id'
    grouped_df = df.groupby('session_id')

    combined_data = pd.DataFrame()

    # Loop through all the grouped data
    for session_id, session_data in grouped_df:
        # Add the session_id as a new column to session_data
        session_data['session_id'] = session_id
        
        # Append session_data to combined_data
        combined_data = pd.concat([combined_data, session_data])

    # Sort the combined data by 'session_id' and 'timestamp'
    combined_data_sorted = combined_data.sort_values(by=['session_id', 'timestamp'], ascending=[True, True])

    # Return the grouped DataFrame after all data is processed
    return combined_data_sorted


# Function to fetch all events and their descriptions for the selected component
def fetch_events_and_descriptions(component_identifier):
    component_ref = db.collection('components').document(component_identifier)
    
    # Fetch all events under the component
    events = component_ref.collection('events').stream()

    # Create a list to store event names and descriptions
    event_list = []

    for event in events:
        event_name = event.id  # Event name
        event_data = event.to_dict()  # Fetch event description
        event_description = event_data.get('event_description', 'No description available')

        # Append the event and description to the list
        event_list.append(f"{event_name}: {event_description}")

    # Join the list into a single string for display
    return "\n".join(event_list)


def generateSimulationData():

    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0, api_key=st.session_state["OPENAI_API_KEY"])
    prompt = f"""
    For a dataframe with the following columns and description that stores event captures
    from react components:
    {st.session_state["events_and_descriptions"]}

    Your job is to write python code to create simulation data. However, there are 
    a few key things to take note:

    1. You create this data from the point of view of the visitor. Therefore, you create
    a column called session_id which uniquely represents a visitors session on the website. 
    Example, if in the same session the user opens the form (form load) and then interacts with the form (button clicked), you
    will create two rows with the same session_id, for the first row the event name will be form load and the value
    will be yes then for the second row the event name will be button click and the value will be clicked. 

    2. For each session_id you create simulation data for the given columns. Your simulation
    data must make sense. For example, form interaction cannot happen without form load. 

    3. The output of your python code must be a pandas dataframe that is stored in a variable
    called generatedData

    Your reply must only be the python code. Do not reply with anything else. Simply reply with 
    the desired python code.
    """
    response = gpt_4o.invoke(prompt).content
    return response


def generateNuggt(context=None):
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0, api_key=st.session_state["OPENAI_API_KEY"])
    
    prompt = f"""
    You are an expert in product analytics for startups. You analyse a given react component
    code and look into the events the component captures to understand how users are interacting
    with that react component. Based on the captured events (user behaviour) data you
    derive possible UI/UX iterations that the product team should consider in order to improve the 
    UI/UX of the react component such that users interact with the component in a manner that is intended
    by the product team. 

    Following is the react component code:
    {st.session_state["fetched_code"]}

    Following are the captured events and their description:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Following is a glimpse of the captured event data:
    {st.session_state["data"]}\n\n"""
    
    if context != None:
        prompt += f"Following are some decisions that the user wants to take:\n\n{context} + \n\n"

    prompt += f"""Based on the provided information, derive as many UI/UX iterations as possible in the following JSON format:

    [
        {{
            "Heading": "<heading of this iteration>",
            "One Metric To Measure": "<metric that will drive decision making for this iteration>",
            "Metric Formula": "<formula for the metric in terms of arithmetic operations on different events captured>",
            "Decision Tree": "<provide different decisions to be made based on different values of the chosen metric, provide three outcomes>",
            "Visualisations": "<describe visualisations to visualise this metric>"
        }},
        // ... more iterations
    ]

    Only reply with the JSON array above and do not include any additional text. Ensure the JSON is valid and properly formatted.
    """
    
    response = gpt_4o.invoke(prompt).content

    try:
        # Parse the JSON response
        nuggts = json.loads(response.replace("```json", "").replace("```", ""))
        
        # Validate that nuggts is a list of dictionaries with required keys
        required_keys = {"Heading", "One Metric To Measure", "Metric Formula", "Decision Tree", "Visualisations"}
        valid_nuggts = []
        for nuggt in nuggts:
            if isinstance(nuggt, dict) and required_keys.issubset(nuggt.keys()):
                valid_nuggts.append((
                    nuggt["Heading"],
                    nuggt["One Metric To Measure"],
                    nuggt["Metric Formula"],
                    nuggt["Decision Tree"],
                    nuggt["Visualisations"]
                ))
            else:
                st.error("Invalid nuggt format detected. Please ensure GPT returns the correct JSON structure.")
                return
        
        # Update the session state with the parsed nuggts
        st.session_state["nuggts"] = valid_nuggts

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}")
        st.text("Response from GPT:")
        st.text(response)
        return

    # Optional: Print the response for debugging purposes
    print(response)

    st.rerun()
    return response

def generate_visuals_code(heading, metric, formula, decision_tree, visualisations):
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    prompt = f"""
    You can reference the event capture data as st.session_state["data"]. 
    Here is a glimpse of the event capture data:
    {st.session_state["data"].to_string()}

    Here is a description of all events captured:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Following is the python code that calculates different metrics, you may reference these metrics by there variable names in your code:
    {st.session_state["tracker"]}

    Based on this your job is to write the python code that: 
    
    1. Generates the following visuals using the plotly.figure_factory python package:
    {visualisations} 

    2. Ensure that all visualisations are properly labelled.
    
    3. Displays all generated visuals using st.plotly_chart.

    4. For each visual, use st.write() to explain how the user should read and interpret the plot.

    Your reply should only contain this python code and nothing else. 
    """

    response = gpt_4o.invoke(prompt).content
    response = response.replace("```python", "").replace("```", "")
    st.session_state["visuals"] = response


def generate_tracker_code(heading, metric, formula, decision_tree, visualisations):
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    prompt = f"""
    You can reference the event capture data as st.session_state["data"]. 
    Here is a glimpse of the event capture data:
    {st.session_state["data"].to_string()}

    Here is a description of all events captured:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Based on this your job is to write the python code: 
    
    1. That extracts the following metric from the st.session_state["data"] pandas dataframe:
    {metric}: {formula}

    2. After extracting the metric, you apply the following conditions to arrive at a decision:
    {decision_tree}

    3. Display the metric value and the corresponding decision using st.write() 

    Your reply should only contain this python code and nothing else. 
    """

    response = gpt_4o.invoke(prompt).content
    response = response.replace("```python", "").replace("```", "")
    st.session_state["tracker"] = response
    print(response)


@st.dialog("Nuggt Editor", width="large")
def edit_nuggt(heading, metric, formula, decision_tree, visualisations):
    info, tracker, visuals, status = st.tabs(["Info", "Tracker", "Visuals", "Status"])
    
    with info:
        st.markdown(f"{heading}")
        st.markdown(f"**One Metric To Measure**: {metric}")
        st.markdown(f"**Metric Formula**: {formula}")
        st.markdown(f"**Decision Tree**: {decision_tree}")
        st.markdown(f"**Visualisations**: {visualisations}")
    
    with tracker:
        code_area = st.container(height=300)
    
        if st.button("Generate Python Code To Track Metric", use_container_width=True):
            generate_tracker_code(heading, metric, formula, decision_tree, visualisations)
            st.session_state["tracker_generated"] = True

        with code_area:
            code_editor(
            code=st.session_state["tracker"],
            lang="python",
            key="first_gen",
            allow_reset=True
        )
    
    with visuals:
        if st.session_state["tracker_generated"]:
            st.session_state["visuals_disabled"] = False
        else:
            st.error("Generate Tracker Code First")

        code_area_visual = st.container(height=300)
    
        if st.button("Generate Python Code For Visuals", use_container_width=True, disabled = st.session_state["visuals_disabled"]):
            generate_visuals_code(heading, metric, formula, decision_tree, visualisations)
            st.session_state["visuals_generated"] = True
            st.session_state["activation_disabled"] = False

        with code_area_visual:
            code_editor(
            code=st.session_state["visuals"],
            lang="python",
            key="visuals",
            allow_reset=True
        )

    with status:
        if not st.session_state["tracker_generated"]:
            st.error("Generate Tracker Code Before Activation")
        else:
            st.success("Tracker Code Generated")
        
        if not st.session_state["visuals_generated"]:
            st.error("Generate Visuals Code Before Activation")
        else:
            st.success("Visuals Code Generated")
        
        card_activation_button = st.button("Activate Card", disabled=st.session_state["activation_disabled"])

        if card_activation_button:
            file_path = "active_cards.txt"

            # Check if the file exists, if not, create it and add the index header
            if not os.path.exists(file_path):
                with open(file_path, 'w') as file:
                    file.write("Index: 1\n")  # Start with index 1 for the first card
                card_index = 1
            else:
                # Read the last index from the file to continue from there
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    last_index = 0
                    for line in lines:
                        if line.startswith("Index:"):
                            last_index = int(line.split(":")[1].strip())
                    card_index = last_index + 1

            # Append the card content with the new index
            with open(file_path, 'a') as file:
                # Write the index
                if card_index > 1:
                    file.write(f"Index: {card_index}\n")
                # Write the content to the file in the specified format
                file.write(f"Heading: {heading}\n")
                file.write(f"One Metric To Measure: {metric}\n")
                file.write(f"Metric Formula: {formula}\n")
                file.write(f"Decision Tree: {decision_tree}\n")
                file.write(f"Visualisations: {visualisations}\n")
                file.write("Tracker Code:\n")
                file.write(f"'''tracker\n{st.session_state['tracker']}\n'''\n")
                file.write("Visuals Code:\n")
                file.write(f"'''visuals\n{st.session_state['visuals']}\n'''\n")
                # Add the separator at the end
                file.write("-----\n")
    

def load_inactive_nuggts():
    """
    Loads nuggts from inactive_cards.txt and returns a list of nuggts.
    Each nuggt is a tuple: (heading, metric, formula, decision_tree, visualisations)
    """
    nuggts = []
    try:
        with open('inactive_cards.txt', 'r') as f:
            content = f.read()
            # Split the content into nuggts using "-----" as separator
            nuggt_texts = content.strip().split("-----")
            for nuggt_text in nuggt_texts:
                nuggt_text = nuggt_text.strip()
                if nuggt_text:
                    nuggt_dict = {}
                    lines = nuggt_text.split('\n')
                    for line in lines:
                        if line.strip():
                            key, value = line.split(':', 1)
                            nuggt_dict[key.strip()] = value.strip()
                    nuggts.append((
                        nuggt_dict.get('heading', ''),
                        nuggt_dict.get('metric', ''),
                        nuggt_dict.get('formula', ''),
                        nuggt_dict.get('decision_tree', ''),
                        nuggt_dict.get('visualisations', '')
                    ))
    except FileNotFoundError:
        # File doesn't exist yet
        pass
    return nuggts

def save_inactive_nuggts(nuggts):
    """
    Saves the list of nuggts to inactive_cards.txt.
    Each nuggt is a tuple: (heading, metric, formula, decision_tree, visualisations)
    """
    with open('inactive_cards.txt', 'w') as f:
        for nuggt in nuggts:
            heading, metric, formula, decision_tree, visualisations = nuggt
            f.write(f"heading: {heading}\n")
            f.write(f"metric: {metric}\n")
            f.write(f"formula: {formula}\n")
            f.write(f"decision_tree: {decision_tree}\n")
            f.write(f"visualisations: {visualisations}\n")
            f.write("-----\n")

def delete_nuggt(index):
    """
    Deletes a nuggt from inactive_cards.txt based on its index.
    """
    nuggts = load_inactive_nuggts()
    if 0 <= index < len(nuggts):
        del nuggts[index]
        save_inactive_nuggts(nuggts)
        st.success("Nuggt deleted successfully.")
        st.rerun()  # Refresh the app to reflect changes
    else:
        st.error("Invalid nuggt index.")

def populateNuggts():
    # Initialize session state variables if not already set
    if "activation_disabled" not in st.session_state:
        st.session_state["activation_disabled"] = True
    if "tracker_generated" not in st.session_state:
        st.session_state["tracker_generated"] = False
    if "visuals_generated" not in st.session_state:
        st.session_state["visuals_generated"] = False
    if "visuals_disabled" not in st.session_state:
        st.session_state["visuals_disabled"] = True
    if "tracker" not in st.session_state:
        st.session_state["tracker"] = "#Python Code to track this metric will appear here"
    if "visuals" not in st.session_state:
        st.session_state["visuals"] = "#Python Code for visualisations will appear here"
    if "nuggts" not in st.session_state:
        st.session_state["nuggts"] = []

    # Save any new nuggts from session state to inactive_cards.txt
    session_nuggts = st.session_state.get("nuggts", [])
    
    # Load existing nuggts from file
    file_nuggts = load_inactive_nuggts()

    if session_nuggts:
        # Append new nuggts from session state
        file_nuggts.extend(session_nuggts)
        # Save updated nuggts list back to file
        save_inactive_nuggts(file_nuggts)
        # Clear the session state nuggts
        st.session_state["nuggts"] = []

    # Load the list of nuggts from inactive_cards.txt
    nuggts = load_inactive_nuggts()
    num_columns = 3  # Define the number of columns per row

    # Initialize index for unique keys and for deletion
    nuggt_idx = 0

    # Iterate over the nuggts in chunks of num_columns
    for i in range(0, len(nuggts), num_columns):
        row = st.columns(num_columns)  # Create a row with 'num_columns' columns

        # Populate each column with nuggt data
        for col in row:
            if nuggt_idx >= len(nuggts):
                break  # No more nuggts to display
            nuggt = nuggts[nuggt_idx]
            heading, metric, formula, decision_tree, visualisations = nuggt

            with col.container(border=True):
                # Display nuggt details
                st.markdown(f"### {heading}")
                st.markdown(f"**One Metric To Measure**: {metric}")
                st.markdown(f"**Metric Formula**: {formula}")
                st.markdown(f"**Decision Tree**\n: {decision_tree}")
                st.markdown(f"**Visualisations**\n: {visualisations}")
                
                # Edit and Activate Card button
                if st.button("Edit and Activate Card", use_container_width=True, key=f"edit_{nuggt_idx}", type="primary"):
                    # Resetting session state variables
                    st.session_state["activation_disabled"] = True
                    st.session_state["tracker_generated"] = False
                    st.session_state["visuals_generated"] = False
                    st.session_state["visuals_disabled"] = True
                    st.session_state["tracker"] = "#Python Code to track this metric will appear here"
                    st.session_state["visuals"] = "#Python Code for visualisations will appear here"

                    edit_nuggt(heading, metric, formula, decision_tree, visualisations)
                
                # Delete Card button
                if st.button("Delete Card", use_container_width=True, key=f"delete_{nuggt_idx}"):
                    delete_nuggt(nuggt_idx)  # Remove nuggt at current index

                nuggt_idx += 1  # Increment nuggt index

def extract_from_file():
    active_cards = []
    with open("active_cards.txt", 'r') as file:
        card = {}
        for line in file:
            if line.startswith("Index:"):
                card["index"] = int(line.split(":")[1].strip())
            elif line.startswith("Heading:"):
                card["heading"] = line.split(":")[1].strip()
            elif line.startswith("One Metric To Measure:"):
                card["metric"] = line.split(":")[1].strip()
            elif line.startswith("Metric Formula:"):
                card["formula"] = line.split(":")[1].strip()
            elif line.startswith("Decision Tree:"):
                card["decision_tree"] = line.split(":")[1].strip()
            elif line.startswith("Visualisations:"):
                card["visualisations"] = line.split(":")[1].strip()
            elif line.startswith("'''tracker"):
                # Read multi-line tracker code
                tracker_code = []
                for code_line in file:
                    if code_line.startswith("'''"):  # End of tracker code block
                        break
                    tracker_code.append(code_line)
                card["tracker_code"] = ''.join(tracker_code)
            elif line.startswith("'''visuals"):
                # Read multi-line visuals code
                visuals_code = []
                for code_line in file:
                    if code_line.startswith("'''"):  # End of visuals code block
                        break
                    visuals_code.append(code_line)
                card["visuals_code"] = ''.join(visuals_code)
            elif line.startswith("-----"):
                active_cards.append(card)
                card = {}  # Reset for the next card
    return active_cards


@tool
def update_tracker_code(instructions:str) -> str:
    """
    Use this tool to update the tracker code. Your input must be detailed instructions
    on what to change. 
    """
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    prompt = f"""
    You can reference the event capture data as st.session_state["data"]. 
    Here is a glimpse of the event capture data:
    {st.session_state["data"].to_string()}

    Here is a description of all events captured:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Following is the existing python code:
    {st.session_state["chosen_tracker_code"]}

    Based on this your job is to:
    {instructions}

    Your reply should only contain this python code and nothing else. 
    """

    response = gpt_4o.invoke(prompt).content
    response = response.replace("```python", "").replace("```", "")
    st.session_state["chosen_tracker_code"] = response
    print(response)

@tool
def update_visual_code(instructions:str) -> str:
    """
    Use this tool to update the visual code. Your input must be detailed instructions
    on what to change. Explicitly mention in your instructions that any new visualisations must be added
    on to the existing visualisations unless stated otherwise by the user. Explicitly mention that
    all visualisations are to be made using plotly.figure_factory and displayed using st.plotly_chart()
    """
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0, api_key=st.session_state["OPENAI_API_KEY"])
    prompt = f"""
    You can reference the event capture data as st.session_state["data"]. 
    Here is a glimpse of the event capture data:
    {st.session_state["data"].to_string()}

    Here is a description of all events captured:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Following is the python code that calculates different metrics, you may reference these metrics by there variable names in your code:
    {st.session_state["chosen_tracker_code"]}

    Following is the existing visualisation code:
    {st.session_state["chosen_visuals_code"]}

    Based on this your job is to:
    {instructions}

    Do not remove current visualisations that are already in the code unless explicitly asked to.
    Always re-write the entire code (including existing code) with ammendments or added visuals.

    Your reply should only contain this python code and nothing else. 
    """

    response = gpt_4o.invoke(prompt).content
    response = response.replace("```python", "").replace("```", "")
    st.session_state["chosen_visuals_code"] = response
    print(response)

@tool
def update_card_description(instructions:str) -> str:
    """
    Use this tool to update the card description. Your input must be detailed instructions
    on what to change.
    """
    gpt_4o = init_chat_model("o1-mini", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    
    prompt = f"""
    You are an expert in product analytics for startups. You analyse a given react component
    code and look into the events the component captures to understand how users are interacting
    with that react component. Based on the captured events (user behaviour) data you
    derive UI/UX iterations that the product team should consider in order to improve the 
    UI/UX of the react component such that users interact with the component in a manner that is intended
    by the product team. 

    Following is the react component code:
    {st.session_state["fetched_code"]}

    Following are the captured events and their description:
    {introducing_session_id + st.session_state["events_and_descriptions"]}

    Following is a glimpse of the captured event data:
    {st.session_state["data"]}
    
    Following is the current state of the UI/UX iteration card:
    Heading: {st.session_state["chosen_heading"]}\n\n
    One Metric To Measure: {st.session_state["chosen_metric"]} 
    Metric Formula: {st.session_state["chosen_formula"]}
    Decision Tree: {st.session_state["chosen_decision_tree"]} 
    Visualisations: {st.session_state["chosen_visualisations"]}\n\n"""
    
    
    prompt += f"Following are the changes to this UI/UX iteration card:\n\n{instructions} + \n\n"

    prompt += f"""Based on the provided instructions, write the updated UI/UX iteration card in the following JSON format:

    [
        {{
            "Heading": "<heading of this iteration>",
            "One Metric To Measure": "<metric that will drive decision making for this iteration>",
            "Metric Formula": "<formula for the metric in terms of arithmetic operations on different events captured>",
            "Decision Tree": "<provide different decisions to be made based on different values of the chosen metric, provide three outcomes>",
            "Visualisations": "<describe visualisations to visualise this metric>"
        }},
    ]

    Only reply with the JSON array above and do not include any additional text. Ensure the JSON is valid and properly formatted.
    """
    
    response = gpt_4o.invoke(prompt).content

    
    # Parse the JSON response
    nuggts = json.loads(response.replace("```json", "").replace("```", ""))
    
    # Validate that nuggts is a list of dictionaries with required keys
    required_keys = {"Heading", "One Metric To Measure", "Metric Formula", "Decision Tree", "Visualisations"}
    valid_nuggts = []
    for nuggt in nuggts:
        if isinstance(nuggt, dict) and required_keys.issubset(nuggt.keys()):
            valid_nuggts.append((
                nuggt["Heading"],
                nuggt["One Metric To Measure"],
                nuggt["Metric Formula"],
                nuggt["Decision Tree"],
                nuggt["Visualisations"]
            ))
        else:
            st.error("Invalid nuggt format detected. Please ensure GPT returns the correct JSON structure.")
            return
    
    # Update the session state with the parsed nuggts
    st.session_state["chosen_heading"] = valid_nuggts[0][0]
    st.session_state["chosen_metric"] = valid_nuggts[0][1]
    st.session_state["chosen_formula"] = valid_nuggts[0][2]
    st.session_state["chosen_decision_tree"] = valid_nuggts[0][3]
    st.session_state["chosen_visualisations"] = valid_nuggts[0][4]



def initialise_agent():
    sys_prompt = f"""
    You are an expert in managing decision cards. Decision cards are basically used
    to improve UI/UX of react components using event capture data. The goal of a decision
    card is to make a decision that is based on data from react component capture data. The goal
    of the decision is to make the react component code better to get the desired behaviour
    out of the user. 

    Following is the current state of the decision card:
    Heading: {st.session_state["chosen_heading"]}\n\n
    One Metric To Measure: {st.session_state["chosen_metric"]} 
    Metric Formula: {st.session_state["chosen_formula"]}
    Decision Tree: {st.session_state["chosen_decision_tree"]}
    Visualisations: {st.session_state["chosen_visualisations"]}

    Following is a description of the captured events from the react component:
    {st.session_state["events_and_descriptions"]}

    We can only use metrics that are can be derived from arithmetic operations on captured events.
    When you update the decision card, you must update the relevant codes as well. Similarly, when you
    update a code, you must update the decision card.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    model = ChatOpenAI(model="gpt-4o")
    tools = [update_tracker_code, update_card_description, update_visual_code]
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

if st.session_state["fetch_component"]:
    # Fetch all component identifiers when the dashboard loads
    component_identifiers = fetch_all_component_identifiers()

# Display the selectbox for component identifiers and store the selection in session state
selected_component = st.selectbox("Select a Component Identifier", component_identifiers, key="component")

# If a component identifier is selected, display the data in tabs
if selected_component:
    if st.button("Fetch Data"):
        st.session_state["fetched_code"] = fetch_component_code(selected_component)
        st.session_state["events_and_descriptions"] = fetch_events_and_descriptions(selected_component)
        st.session_state["data"] = fetch_and_group_data_by_session(selected_component)
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["About", "Raw Data", "Agent", "Inactive Decision Cards", "Active Decision Cards"])

    with tab1:
        st.subheader("Events and Descriptions")
        st.text(st.session_state["events_and_descriptions"])  # Display the formatted events and descriptions
        
        st.subheader("Component Code")
        code_editor(
            code=st.session_state["fetched_code"],
            lang="typescript",
            key="fetched_code",
            allow_reset=True
        )
        
    with tab2:
        st.subheader("Raw Data (Per Session)")
        data = st.session_state["data"]
        if not data.empty:
            st.dataframe(data[['session_id', 'timestamp', 'event_name', 'value', 'value_type', 'event_description']])

    with tab3:
        st.header("Decision Card Agent")

        if st.session_state["fetched_code"] != "Code not fetched":
            # Load cards from file
            cards = extract_from_file()

            # Create a selector for the card index and heading
            card_options = [(f"{card['index']} - {card['heading']}", card) for card in cards]
            selected_card_option = st.selectbox("Select a card", card_options, format_func=lambda option: option[0])

            if selected_card_option:
                if st.button("Select Card"):
                    # Store the selected card's details in session state
                    selected_card = selected_card_option[1]
                    st.session_state["chosen_index"] = selected_card["index"]
                    st.session_state["chosen_heading"] = selected_card["heading"]
                    st.session_state["chosen_metric"] = selected_card["metric"]
                    st.session_state["chosen_formula"] = selected_card["formula"]
                    st.session_state["chosen_decision_tree"] = selected_card["decision_tree"]
                    st.session_state["chosen_visualisations"] = selected_card["visualisations"]
                    st.session_state["chosen_tracker_code"] = selected_card["tracker_code"]
                    st.session_state["chosen_visuals_code"] = selected_card["visuals_code"]
                    st.session_state["agent_executor"] = initialise_agent()
                    st.session_state["chat_history"] = []

                col1, col2, col3 = st.columns([1.5, 2, 2])
                with col1:
                    with st.container(height=500):
                        # Display the selected card details
                        st.write("### Selected Card Details")
                        st.write(f"**Index:** {st.session_state['chosen_index']}")
                        st.write(f"**Heading:** {st.session_state['chosen_heading']}")
                        st.write(f"**One Metric To Measure:** {st.session_state['chosen_metric']}")
                        st.write(f"**Metric Formula:** {st.session_state['chosen_formula']}")
                        st.write(f"**Decision Tree:** {st.session_state['chosen_decision_tree']}")
                        st.write(f"**Visualisations:** {st.session_state['chosen_visualisations']}")
                
                with col2:
                    with st.container(height=500):
                        col1, col2 = st.columns([1.5, 1])
                        with col1:
                            st.write(f"### Tracker Code:")
                        with col2:
                            tracker_tab = ui.tabs(options=['Preview', 'Code'], default_value='Code', key="test")
                        if tracker_tab == "Code":
                            st.code(st.session_state["chosen_tracker_code"])
                        else:
                            exec(st.session_state['chosen_tracker_code'])

                with col3:    
                    with st.container(height=500):
                        col1, col2 = st.columns([1.5, 1])
                        with col1:
                            st.write(f"### Visuals Code:")
                        with col2:
                            visual_tab = ui.tabs(options=['Preview', 'Code'], default_value='Code', key="test2")
                        if visual_tab == "Code":
                            st.code(st.session_state['chosen_visuals_code'])
                        else:
                            exec(st.session_state['chosen_visuals_code'])
                
                with st.container(border=True):
                    with st.chat_message("assistant"):
                        st.write(st.session_state["current_response"])

                if prompt := st.chat_input("Message Decision Card Agent"):
                    st.session_state["chat_history"].append(HumanMessage(prompt))
                    st.session_state["current_response"] = st.session_state["agent_executor"].invoke({"input": prompt, "chat_history": st.session_state["chat_history"]})["output"]
                    st.session_state["chat_history"].append(AIMessage(st.session_state["current_response"]))
                    st.rerun()
        else:
            st.write("Please fetch data first.")

    with tab4:
        st.header("Inactive Cards")
        if st.session_state["fetched_code"] != "Code not fetched":
            context = st.text_area("Enter any relevant context here (llm will generate random cards based on data if left empty)")
            if st.button("Generate Random Cards", use_container_width=True):
                if context != "":
                    generateNuggt(context)
                else:
                    generateNuggt()
            st.markdown("---")
            populateNuggts()
        else:
            st.write("Please fetch data first.")

    with tab5:
        st.header("Active Cards")
        if st.session_state["fetched_code"] != "Code not fetched":
            if not os.path.exists("active_cards.txt"):
                st.write("No Active Cards")
            else:
                active_cards = extract_from_file()  # Fetch cards with index
                for idx, entry in enumerate(active_cards):
                    # Expandable section to display and execute the tracker code and visuals code
                    with st.expander(f"## Card {entry['index']}: {entry['heading']}"):
                        # Markdown part to display the heading, metric, formula, decision tree, and visualisations
                        st.markdown(f"### {entry['heading']}")
                        st.markdown(f"**One Metric To Measure**: {entry['metric']}")
                        st.markdown(f"**Metric Formula**: {entry['formula']}")
                        st.markdown(f"**Decision Tree**: {entry['decision_tree']}")
                        st.markdown(f"**Visualisations**: {entry['visualisations']}")
                        
                        # Tracker code execution within its own container
                        with st.container():
                            st.markdown("#### Executing Tracker Code")
                            try:
                                exec(entry['tracker_code'])
                            except Exception as e:
                                error_message = traceback.format_exc()
                                st.error(f"Error executing tracker code:\n```\n{error_message}\n```")

                        # Visuals code execution within its own container
                        with st.container():
                            st.markdown("#### Executing Visuals Code")
                            try:
                                exec(entry['visuals_code'])
                            except Exception as e:
                                error_message = traceback.format_exc()
                                st.error(f"Error executing visuals code:\n```\n{error_message}\n```")

        else:
            st.write("Please fetch data first.")

    
   
            
