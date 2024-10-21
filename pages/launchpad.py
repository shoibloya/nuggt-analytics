import streamlit as st
import os
import re
from code_editor import code_editor  # Import code editor module
from openai import OpenAI
from google.cloud import firestore
import json
import os
from langchain.chat_models import init_chat_model
import streamlit_shadcn_ui as ui
import pandas as pd
import streamlit.components.v1 as components
import subprocess
import psutil
import time

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'firebase/serviceAccountKey.json'

if 'populated' not in st.session_state:
    st.session_state['populated'] = False

if 'tracking_data' not in st.session_state:
    st.session_state['tracking_data'] = []

if 'current_change' not in st.session_state:
    st.session_state['current_change'] = None

if 'from_ai' not in st.session_state:
    st.session_state['from_ai'] = False

if 'component' not in st.session_state:
    st.session_state['component'] = ""

# Set the page configuration
st.set_page_config(
    page_title="Analytics Setup",
    layout="wide",
    initial_sidebar_state="collapsed"
)

db = firestore.Client()

# Set the root directory
ROOT_DIR = 'uploaded_projects'

# Define allowed React file extensions
ALLOWED_EXTENSIONS = ['.js', '.jsx', '.ts', '.tsx']

def is_allowed_file(filename):
    """Check if the file has an allowed React extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def parse_tracking_json(json_text):
    """
    Parse the JSON text string into a Python dictionary.

    Args:
        json_text (str): The JSON text string.

    Returns:
        dict: The parsed JSON data.
    """
    return json.loads(json_text)

def get_react_files(directory):
    """
    Recursively traverse the directory and return a list of paths to React component files,
    excluding any files under 'node_modules' directories.

    Args:
        directory (str): The root directory to search.

    Returns:
        List[str]: A list of file paths relative to ROOT_DIR.
    """
    react_files = []
    for root, dirs, files in os.walk(directory):
        # Exclude 'node_modules' and '.next' directories
        dirs[:] = [d for d in dirs if d.lower() not in ['node_modules', '.next']]

        for file in files:
            if is_allowed_file(file):
                # Get the relative path
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                react_files.append(relative_path)
    return react_files

def kill_process_and_children(pid):
    """
    Kill a process and all of its child processes.

    Args:
        pid (int): The process ID of the parent process.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def wait_for_server(url, timeout=30):
    """
    Wait until the server at the given URL is responsive.

    Args:
        url (str): The URL to check.
        timeout (int): The maximum time to wait in seconds.

    Returns:
        bool: True if the server is responsive, False if timed out.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except:
            pass
        if time.time() - start_time > timeout:
            return False
        time.sleep(0.5)

def run_npm_in_subfolder():
    # Get the path to 'uploaded_projects'
    initial_cwd = os.getcwd()
    uploaded_projects_path = os.path.join(initial_cwd, 'uploaded_projects')

    # Step 1: Check if 'uploaded_projects' exists
    if not os.path.exists(uploaded_projects_path):
        print("uploaded_projects directory does not exist")
        return

    # Step 2: Find the subfolder name
    subfolders = [f.name for f in os.scandir(uploaded_projects_path) if f.is_dir()]

    if not subfolders:
        print("No subfolders found in uploaded_projects")
        return

    # Step 3: Get the path to the first subfolder
    subfolder = subfolders[0]  # Assuming you want the first subfolder
    subfolder_path = os.path.join(uploaded_projects_path, subfolder)

    # Kill any existing 'npm run dev' process
    if 'npm_process' in st.session_state and st.session_state['npm_process'] is not None:
        pid = st.session_state['npm_process'].pid
        kill_process_and_children(pid)
        st.session_state['npm_process'] = None

    # Step 4: Run 'npm run dev' as a background process with PORT=3000
    try:
        # 'npm run dev' command with PORT=3000 as a background process
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            env={**os.environ, "PORT": "3000"},
            cwd=subfolder_path
        )
        st.session_state['npm_process'] = process
        # Give the server some time to start
        time.sleep(10)  # Adjust the sleep time if needed
    except Exception as e:
        print(f"Error while running npm: {e}")

# Streamlit function to set up Firestore structure with a DataFrame
def setup_component_structure(component_identifier, react_component_code, events_df):
    # Set up Firestore document for the component identifier
    doc_ref = db.collection('components').document(component_identifier)

    # Store the React component code in the document
    doc_ref.set({
        'react_code': react_component_code
    })

    # Set up the event structure under the 'events' sub-collection
    for index, row in events_df.iterrows():
        event_name = row['Event Name']
        event_description = row['Event Description']

        # Create a document for each event in the 'events' sub-collection
        event_ref = doc_ref.collection('events').document(event_name)
        event_ref.set({
            'event_name': event_name,
            'event_description': event_description
        })

    st.success(f"Component '{component_identifier}' and its events have been successfully set up in Firestore!")

def updateChatbotContext():
    sys_prompt = f"""

    You are an expert in data driven UI/UX decisions. Given a react component code, you
    help users identify the events they can keep track of in order to make data driven
    UI/UX iterations that will eventually improve the components user experience. 

    Following is the react component code: 

    {file_content}

    Here are the current things the user intends to track:

    {str(st.session_state["tracking_data"])}

    Your job today is to interact with the user in following JSON format:

    ```json
    {{
        "reply": "Your reply to the user summarising your actions or answering their queries",
        "tracking": [ //You can leave this part empty if there is nothing to add
        {{ 
            "events": [
                {{
                    "event_name": "new event name or existing event name to edit",
                    "event_description": "What the new or updated event captures"
                }},
                // And so on if you want to add more than one thing
        ]
        }}
            
        ]
    }}
    ```

    Here are a few examples: 

    Based on user's query, if you want to add a new event called form_duration:

    ```json
    {{
        "reply": "I have added form_duration",
        "tracking": [ 
        {{ 
            "events": [
                {{
                    "event_name": "form_duration",
                    "event_description": "tracks how long the form was open for"
                }}
                // And so on for to add more
        ]
        }}
        ]
    }}
    ```

    Based on user's query, if you want to edit an existing event called input_change, you will reply:

    ```json
    {{
        "reply": "I have edited input_change",
        "tracking": [ 
        {{
            "events": [
                {{
                    "event_name": "input_change",
                    "event_description": "updated input_change keeps track of the value change"
                }}
                // And so on for other edits
        ]
        }}
        ]
    }}
    ```

    If the user is asking you a question and does not intend for your to make any changes to the tracking list, you will reply in the following manner:

    ```json
    {{
        "reply": "Your reply to user's query",
        "tracking": "Not applicable"
    }}
    ```
    """
    st.session_state.messages = [{"role": "system", "content": sys_prompt}]

def getData(code):
    gpt_4o = init_chat_model("gpt-4o", model_provider="openai", temperature=0, api_key=st.session_state["OPENAI_API_KEY"])
    
    prompt = f'''
    You are an expert in user behavior analytics and UI/UX optimization.

    Your task:

    - Analyze the following React component code:

    \`\`\`tsx
    {code}
    \`\`\`

    - Suggest as many events that can be tracked in the given react code (without implementing any new elements) in the following JSON format:

    \`\`\`json
        {{
        "tracking": [
            {{
            "events": [
                {{
                "event_name": "Event name",
                "event_description": "Describe the event in detail."
                }}
                // And so on..
            ]
            }}
        ]
        }}
    \`\`\`

Only provide the JSON in your response. Do not include any additional text.
    '''
    data = gpt_4o.invoke(prompt).content
    data = data.replace("```json", "").replace("```", "")
    return data

def integrateAnalytics(current_code=None):
    things_to_track = ""
    gpt_4o = init_chat_model("o1-preview", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    for index, row in st.session_state['df'].iterrows():
        things_to_track = things_to_track + "\nEvent Name: " + row['Event Name'] + "\nEvent Description: " + row['Event Description'] + "\n\n"
    prompt = f"""
    You are an analytics expert for react components. Given a component code and a list
    of events to capture, you re-write the complete code such that it tracks all the events
    in the list and console.log() the values of the captured events. 

    Following is the react component code tsx:
    {current_code}

    Following are a list of things to track:
    {things_to_track}

    Reply with the updated react component code. Do not reply or explain anything else.
    """
    response = gpt_4o.invoke(prompt).content
    
    st.session_state["component"] = response.replace("```tsx", "").replace("```", "")

    sys_prompt_code = f"""
    You are an analytics expert for react components. Given a component code and a list
    of events to capture, you re-write the complete code such that it tracks all the events
    in the list and console.log() the values of the captured events.

    These are the list of things to track:
    {things_to_track}

    If the user asks you to track things beyond this list, you are kind enough to help
    them track events as they need. 

    If based on the user's query you realise that there are changes to the code, 
    please reply in the following format:

    'Explanation: <This is where you explain the changes that you are making>
    Code: <This is where you re-write the enture typescript code with the relevant
    changes such that the user can simply copy-paste your code and it works>'

    If based on the user's query you realise that there are no changes to the code,
    please reply in the following manner:

    'Response: <Answer user's query>
    Code: NA'
    """
    st.session_state.cmessages.append({"role": "system", "content": sys_prompt_code}) 
    
def replace_file_content_with_component(file_path):
    # Check if 'component' exists in session state
    if "component" in st.session_state:
        # Get the component content from the session state
        component_content = st.session_state["component"]
        
        try:
            # Open the file in write mode to overwrite its content
            with open(file_path, 'w') as file:
                file.write(component_content)
            st.success(f"File {file_path} updated successfully!")
        except Exception as e:
            st.error(f"An error occurred while writing to the file: {e}")
    else:
        st.warning("No component found in session state.")

def getComponents(from_generate=False, from_ai=False, **kwargs):
    # Check if 'df' (the dataframe) exists in session state, if not, initialize it
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame(columns=["Event Name", "Event Description"])

    if from_ai:
        change = kwargs.get('change', None)
        if change:  # Ensure 'change' is not None and has elements
            change = change[0]
            events = change.get('events', [])
            for event in events:
                event_name = event.get('event_name')
                event_description = event.get('event_description')

                # Update the existing table in session_state
                if event_name in st.session_state['df']["Event Name"].values:
                    st.session_state['df'].loc[
                        st.session_state['df']["Event Name"] == event_name, "Event Description"
                    ] = event_description
                else:
                    # Append new row if the event_name is not already in the table
                    new_row = pd.DataFrame(
                        [{"Event Name": event_name, "Event Description": event_description}]
                    )
                    st.session_state['df'] = pd.concat([st.session_state['df'], new_row], ignore_index=True)
    elif from_generate:
        # Initialize the table with tracking data for the first time
        data = []
        for idx, tracking_item in enumerate(st.session_state['tracking_data']['tracking']):
            events = tracking_item['events']
            for event_idx, event in enumerate(events):
                data.append({"Event Name": event['event_name'], "Event Description": event['event_description']})

        st.session_state['df'] = pd.DataFrame(data)

    # Display the dataframe using st.data_editor in both cases
    st.session_state['df'] = st.data_editor(st.session_state['df'], num_rows="dynamic", use_container_width=True)

@st.dialog("File Edit Warning")
def edit_warning():
    st.write(f"The original file will be overwritten")
    if st.button("Continue"):
        st.session_state["warning_accepted"] = True
        st.rerun()

def integrateFirebase(component_identifier):
    gpt_4o = init_chat_model("o1-preview", model_provider="openai", temperature=1, api_key=st.session_state["OPENAI_API_KEY"])
    things_to_track = ""

    for index, row in st.session_state['df'].iterrows():
        things_to_track = things_to_track + "\nEvent Name: " + row['Event Name'] + "\nEvent Description: " + row['Event Description'] + "\n\n"

    prompt = f"""
    You are an expert in integrating firebase into react component code such that all
    captured events are sent to firestore. In order to send captured events to firestore
    you inject the following code as it is in the react component code:

    import {{ db }} from '../lib/firebase';
    import {{ useSession }} from '@/contexts/SessionContext';
    
    //define the following function
    const logEventToFirestore = async (eventName, sessionId, extraData = {{}}) => {{
    try {{
      const eventRef = doc(db, 'components', {component_identifier}, 'events', eventName);
      await addDoc(collection(eventRef, 'data'), {{
        timestamp: Date.now(),
        session_id: sessionId,
        ...extraData
      }});
      console.log(`Event '{{eventName}}' logged to Firestore with session ID {{sessionId}}`);
    }} catch (e) {{
      console.error('Error logging event to Firestore: ', e);
    }}
    }};

    You also write code to generate a newSessionId when the user enters the page using the following code:
    const newSessionId = useSession();  // Capture the global session_id
    setSessionId(newSessionId);

    Here is an example of how you convert events captured in console.log to be sent to firestore:
    logEventToFirestore(<event_name>, newSessionId);

    Be extremely sure that you use the correct event name because event name is crucial to ensure
    that the data is stored at the correct place on firestore. Following is the list of event names
    and descriptions:

    {things_to_track}

    Following is the react component code tsx:
    {st.session_state["component"]}

    Reply with the updated react component code. Do not reply or explain anything else.
    """
    response = gpt_4o.invoke(prompt).content
    
    st.session_state["component"] = response.replace("```tsx", "").replace("```", "")

st.title("Configure Analytics for React Component")
tab1, tab2, tab3, tab4 = st.tabs(["Select React Component", "Events To Capture", "Integrate Analytics Code", "Connect To Firestore"])

with tab1:
    # Check if ROOT_DIR exists
    if not os.path.exists(ROOT_DIR) or not os.path.isdir(ROOT_DIR):
        st.error(f"The directory **{ROOT_DIR}** does not exist.")
    else:
        # Get all React component files
        react_files = get_react_files(ROOT_DIR)

        if not react_files:
            st.warning(f"No React component files found in **{ROOT_DIR}** (excluding node_modules).")
        else:
            # Dropdown for selecting a React component file
            st.subheader("Step 1: Select a React Component File")
            selected_file = st.selectbox(
                "Choose a file:",
                options=react_files,
                index=0,
                help="Select a React component file to view and edit its content."
            )

            # Full path to the selected file
            file_path = os.path.join(ROOT_DIR, selected_file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Display the filename
                st.subheader(f"📄 {selected_file}")

                # Integrate code editor for file content
                edited_code = code_editor(
                    code=file_content,
                    lang="typescript",
                )

            except Exception as e:
                st.error(f"Error reading the file: {e}")

with tab2:
    col1, col2 = st.columns([8, 2])

    with col1:
        # Step 2: Decide what to track
        st.subheader("Step 2: Decide what to track")

    with col2: 
        generateButton = st.button("Generate Tracking Suggestions")

    # Divide into two columns
    col1, col2 = st.columns([4, 6])

    with col1:
        with st.container(height=600):
            history = st.container(height=500)
            client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])  # Replace with your OpenAI API key

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with history:
                        with st.chat_message(message["role"]):
                            if message["role"] == "assistant":
                                to_append = parse_tracking_json(message["content"])["reply"]
                                st.markdown(to_append)
                            elif message["role"] == "user":
                                st.markdown(message["content"])

            if prompt := st.chat_input("What is up?"):
                template = f"""Reply to the following user query in the required JSON format:

                {prompt}
                """

                st.session_state.messages.append({"role": "user", "content": prompt})
                with history:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with history:
                    with st.chat_message("assistant"):
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=False,
                            temperature=0,
                        )
                        response = response.choices[0].message.content
                        response = response.replace("```json", "").replace("```", "")
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        response = parse_tracking_json(response)
                        if response["tracking"] != 'Not applicable':
                            st.session_state['from_ai'] = True
                            st.session_state['current_change'] = response["tracking"]
                        st.write(response["reply"])

    with col2:
        if generateButton:
            st.session_state['populated'] = True
            st.session_state['from_ai'] = False
            st.session_state['current_change'] = None
            st.session_state['tracking_data'] = parse_tracking_json(getData(file_content))
            updateChatbotContext()
            getComponents(from_generate=True)
        elif st.session_state['from_ai']:
            st.session_state['from_ai'] = False
            getComponents(from_ai=True, change=st.session_state['current_change'])
        elif st.session_state['populated']:        
            getComponents()


with tab3:

    col1, col2, col3 = st.columns([7, 1.5, 1.5])

    with col1:  
        st.subheader("Step 3: Integrate Analytics to component")
        
    with col2: 
        generateCode = st.button("Integrate Analytics")
        
    with col3:
        test = ui.tabs(options=['Preview', 'Code'], default_value='Code', key="test")

    with st.container(height=650):
        if test == "Code":
            col1, col2 = st.columns([4, 6])
            with col1:
                with st.container(height=600):
                    history = st.container(height=500)
                    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])  # Replace with your OpenAI API key

                    if "cmessages" not in st.session_state:
                        st.session_state.cmessages = []

                    for message in st.session_state.cmessages:
                        if message["role"] != "system":
                            with history:
                                with st.chat_message(message["role"]):
                                    if message["role"] == "user":
                                        st.markdown(message["content"].split("User's query:")[1])
                                    else:
                                        st.markdown(message["content"].split("Code:")[0])

                    if prompt := st.chat_input("What is up?", key="codeInput"):
                        template = f"""Reply to the following user query in the required format:

                        Current State of Code:
                        {st.session_state["component"]}

                        User's query:
                        {prompt}
                        """

                        st.session_state.cmessages.append({"role": "user", "content": template})
                        with history:
                            with st.chat_message("user"):
                                st.markdown(prompt)

                        with history:
                            with st.chat_message("assistant"):
                                response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.cmessages
                                    ],
                                    stream=False,
                                    temperature=0,
                                )
                                response = response.choices[0].message.content
                                parsed_response = response.split("Code:")
                                explanation = parsed_response[0]
                                code_match = re.search(r"```typescript([\s\S]*?)```", parsed_response[1])
                                
                                if code_match:
                                    st.session_state["component"] = code_match.group(1).strip()
                                    
                                st.session_state.cmessages.append({"role": "assistant", "content": response})
                                st.write(explanation.replace("Explanation:", "").replace("Response:", ""))

            with col2:

                if 'initial_integration' not in st.session_state:
                    st.session_state["initial_integration"] = False

                if generateCode:
                    integrateAnalytics(file_content)

                    st.session_state["initial_integration"] = True
                    with st.container(height=600):
                        code_editor(
                                code=st.session_state["component"],
                                lang="typescript",
                                key="first_gen",
                                allow_reset=True
                            )

                elif st.session_state["initial_integration"]:
                    with st.container(height=600):
                        code_editor(
                                code=st.session_state["component"],
                                lang="typescript",
                                key="subs_gen",
                                allow_reset=True
                        )

        else:
            if "warning_accepted" not in st.session_state:
                st.session_state["warning_accepted"] = False
            
            if st.session_state["warning_accepted"]:
                st.session_state["warning_accepted"] = False
                replace_file_content_with_component(file_path)
                run_npm_in_subfolder()
                components.iframe("http://localhost:3000/", height=1000, scrolling=True)
            else:
                edit_warning()


with tab4:
            
    st.subheader("Step 4: Creating a Firestore collection")
    component_identifier = st.text_input("Enter Component Identifier")

    # Set up the Firestore structure when button is pressed
    if st.button("Set Up Component in Firestore"):
        if component_identifier:
            setup_component_structure(component_identifier, st.session_state["component"], st.session_state["df"])
        else:
            st.error("Please fill out all fields!")

    if st.button("Integrate Firebase to Code"):
        integrateFirebase(component_identifier)
        code_editor(
                    code=st.session_state["component"],
                    lang="typescript",
                    key="firebase",
                    allow_reset=True
            )



