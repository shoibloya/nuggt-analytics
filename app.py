import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Instructions",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#st.session_state["OPENAI_API_KEY"] = "<your-api-key>"

# Title for the Instructions page
st.title("📋 Instructions")

# Instructions content
st.markdown("""
# Overview

This page provides a step-by-step guide on how to use the app. The process can be broken down into three main steps:

1. **Setup**: The first step is to complete the initial setup of the app. This involves configuring your environment and making sure all necessary dependencies are installed and properly set up.

2. **Connecting Analytics to React Code**: Once the setup is complete, the next step is to integrate analytics tracking into your React code. This allows you to capture user interactions and send data to the backend for analysis.

3. **Generating Decision Cards and Visualizations**: Finally, after your analytics are connected, you can use the dashboard to generate decision cards and visualizations. This is powered by an LLM agent that will provide insights and visual data to help you make informed decisions based on user behavior.
""")

st.markdown("""
## Step 1: Setup

To get started with the app, follow these initial setup steps:

1. **Set up OpenAI API Key**:
    - Navigate to `app.py` in your project directory.
    - In this file, you need to add your OpenAI API key to ensure the app can interact with the language model.
    - Locate the section of the code where the API key is set, and update it as follows:

    ```python
    openai.api_key = "your-openai-api-key"
    ```

    Make sure to replace `"your-openai-api-key"` with your actual API key obtained from OpenAI.

2. **Update Firebase Service Account Key**:
    - You will also need to configure Firebase for the app to store and retrieve data correctly.
    - Go to the `firebase` folder in your project, and locate the file `serviceAccountKey.json`.
    - Replace the contents of this file with the service account credentials from your Firebase project. You can download the service account key from your Firebase console under **Project Settings** > **Service Accounts** > **Generate New Private Key**.

3. **Add Your Next.js Project**:
    - **Upload Next.js Project**:
        - Place your Next.js project inside the `uploaded_projects` directory in the app folder.
    
    - **Session Management**:
        - There’s no need to create `SessionWrapper.tsx` or `SessionContext.tsx`, as both files are already available in the repository.
    
    - **Set up `firebase.js` under `lib`**:
        - To integrate Firebase, create a new file called `firebase.js` in your `lib` folder.
        - Add the Firebase configuration and initialization logic. Here’s an example of how to set it up:
        
        ```javascript
        import { initializeApp } from 'firebase/app';
        import { getFirestore } from 'firebase/firestore';
        
        const firebaseConfig = {
            apiKey: 'your-firebase-api-key',
            authDomain: 'your-firebase-auth-domain',
            projectId: 'your-firebase-project-id',
            storageBucket: 'your-firebase-storage-bucket',
            messagingSenderId: 'your-messaging-sender-id',
            appId: 'your-app-id',
        };
        
        const app = initializeApp(firebaseConfig);
        const db = getFirestore(app);
        
        export { db };
        ```

    - **Wrap Components with `SessionWrapper`**:
        - In your `layout.tsx`, wrap the `{children}` with `SessionWrapper` to ensure session management is applied across your app.
        - Update `layout.tsx` as follows:
        
        ```tsx
        import SessionWrapper from '@/components/SessionWrapper';
        
        export default function Layout({ children }) {
            return (
                <SessionWrapper>
                    {children}
                </SessionWrapper>
            );
        }
        ```

4. **Ensure the App Runs with `npm run dev`**:
    - After completing the setup, ensure that the app can run locally.
    - Open your terminal and navigate to your project directory.
    - Run the following command:

    ```bash
    npm run dev
    ```

    This will start the development server. Make sure everything is working as expected and there are no errors.
""")

st.markdown("""
## Step 2: Connecting Analytics to React Code

Follow these steps to connect analytics tracking to your React components:

1. **Go to the Launchpad Page**:
    - Start by navigating to the Launchpad page in the app.
    - Select the React component you want to track.

2. **Generate Tracking Suggestions**:
    - Click on the **Events to Capture** tab and then click the **Generate Tracking Suggestions** button.
    - You can interact with the chatbot to make any changes to the suggested tracking events, or directly edit the table as needed.

3. **Integrate Analytics Code**:
    - Once you're satisfied with the tracking events, go to the **Integrate Analytics Code** tab.
    - Click the **Integrate Analytics** button. This will integrate the analytics code to capture events in your React component and log them to the console.
    - To check if the integration works, click on **Preview** and open the browser console to see if the events are being captured.

4. **Interact with the Chatbot**:
    - If you need to make changes to the code or fix any errors, you can interact with the chatbot in the Launchpad to make adjustments to the analytics code.

5. **Connect to Firestore**:
    - After verifying the analytics integration, go to the **Connect to Firestore** tab.
    - Give your component a unique ID and click on **Set up Component in Firestore**. This will create the necessary Firestore storage for your component.
    - Finally, click on the **Integrate Firebase to Code** button. This will modify the event capture code to send the captured events to Firebase instead of just logging them to the console.
""")

st.markdown("""
## Step 3: Generating Decision Cards and Visualizations

**Disclaimer**: Before proceeding to the dashboard, ensure you have generated some data by interacting with your app. Make sure that the generated data contains at least a few session IDs by refreshing the page to capture multiple sessions.

1. **Fetch Data**:
    - First, select the component that is connected to Firebase from the dropdown in the dashboard.
    - Click on the **Fetch Data** button to retrieve the raw data for that component.

2. **Generate Inactive Decision Cards**:
    - Once you have the raw data, navigate to the **Inactive Decision Cards** tab.
    - Click on **Generate Cards** to get a set of decision cards. These cards evaluate key metrics and suggest potential UI/UX changes and other decisions that you can implement for your component.
    - The cards are self-explanatory, and they give actionable insights based on your metrics. 

3. **Activate Decision Cards**:
    - After generating the decision cards, you can activate any of them. Once a card is activated, it will appear under the **Active Cards** section. 
    - Remember to refresh the page to see the activated cards.

4. **Interact with the Agent**:
    - In the **Agent** tab, you can choose any active decision card and interact with the LLM agent.
    - The agent can assist you in making changes, discussing the card’s insights, or even generating code for implementing the suggested actions. This makes it easy to fine-tune your component's performance and user experience.
""")


