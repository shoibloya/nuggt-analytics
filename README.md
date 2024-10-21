# Nuggt: An LLM Agent that Runs on React Component Event Capture Data

1. [Join Discord for issues and troubleshooting](https://discord.gg/4u4ZvTp4)
2. [Follow on X for updates](https://x.com/LoyaShoib)

Nuggt is an LLM-powered agent that helps optimize user experiences by analyzing event data captured in React components. This project integrates with Firebase to store event data, utilizes analytics to track user interactions, and generates decision cards to suggest improvements based on user behavior. The LLM agent can assist in making decisions and generating code based on the insights derived from these decision cards.


https://github.com/user-attachments/assets/8b64d89c-9038-4660-a2ce-3cb44d190639


```bash
# Clone the repository
git clone <your-repo-url>

# Navigate to the project directory
cd <your-repo-directory>

# Install the required dependencies
pip install -r requirements.txt
```

## Overview

This project is broken down into three key steps:

1. **Setup**: Configure your environment and dependencies, such as the OpenAI API key, Firebase service account, and integrating your Next.js project.
2. **Connecting Analytics to React Code**: Set up analytics to track user interactions and capture events in your React components.
3. **Generating Decision Cards and Visualizations**: Use the dashboard to generate decision cards and visualizations to gain insights and take action on improving the UI/UX.

## Step 1: Setup

To get started, follow these steps to set up your environment:

1. **Set up OpenAI API Key**:
    - Add your OpenAI API key to the `app.py` file.
  
2. **Update Firebase Service Account Key**:
    - Replace the Firebase `serviceAccountKey.json` file with your Firebase credentials.

3. **Add Your Next.js Project**:
    - Place your Next.js project in the `uploaded_projects` folder.
    - The `SessionWrapper.tsx` and `SessionContext.tsx` files are already available in the repo.
  
4. **Set up Firebase**:
    - Create a `firebase.js` file under the `lib` folder to configure Firebase and Firestore.
  
5. **Run the App**:
    - Make sure your project can run at:
    
    ```bash
    npm run dev
    ```

## Step 2: Connecting Analytics to React Code

1. Navigate to the **Launchpad** page and select the React component you want to track.
2. Generate tracking suggestions by clicking **Generate Tracking Suggestions** in the **Events to Capture** tab.
3. Integrate analytics by clicking **Integrate Analytics** and verifying the event logs in the browser console.
4. Connect to Firestore and set up Firebase to store captured events by providing a unique ID for your component.

## Step 3: Generating Decision Cards and Visualizations

Before proceeding, ensure you have generated some session data.

1. Fetch data by selecting your Firebase-connected component and clicking **Fetch Data**.
2. Generate decision cards to gain insights and suggestions based on your data metrics.
3. Activate cards to apply the insights, and refresh the page to see active cards.
4. Use the LLM agent to interact with any active decision card, discuss insights, or generate code.

