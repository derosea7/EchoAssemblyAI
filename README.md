Overview
--------

This project is a personal project demonstrating the development of a voice assistant designed to enhance meeting comprehension and engagement. The core functionality centers around real-time transcription, conversation summarization, trend detection, and thought generation. The goal is to provide the user with critical insights and prompts during meetings, minimizing the need for constant attention while still allowing informed participation. This project highlights the practical application of LLMs (Large Language Models) for enhancing productivity and accessibility.

Features
--------

*   **Real-time Transcription:** Utilizes the AssemblyAI API to transcribe meeting audio.
*   **Summarization:** Summarizes segments of the conversation using Google's Vertex AI (Gemini Flash model) to provide context and catch the user up.
*   **Trend Detection:** Analyzes conversation summaries to identify emerging, fading topics, and shifting emphasis within the meeting.
*   **Thought Generation:** Generates potential thoughts and prompts for the user based on conversation summaries and trend analysis, utilizing Cerebras's low-latency Llama model (or other LLMs depending on setup). This is to help the user stay informed and potentially engaged.

Technical Details
-----------------

*   **Language:** Python
*   **APIs/Libraries:**
    *   AssemblyAI (for transcription)
    *   Google Vertex AI (for summarization - Gemini Flash Model)
    *   Cerebras (for low-latency thought generation - Llama Model)
    *   OpenAI (Optional, for LLM flexibility)
*   **Architecture:**
    *   The `runner.py` file is the core program that orchestrates the workflow.
    *   `cerebras_summarizer.py` encapsulates the interaction with different LLMs for summarization and thought generation.
    *   `trend_detection.py` defines a model for understanding the conversation's evolving trends.
    *   `conversation_marker_definitions.py` (currently unused) outlines the concept of tracking elements like questions, answers, and action items.

Setup and Installation
----------------------

1.  **Prerequisites:**
    *   Python (Install if not already present - including a Python environment)
    *   A Git client (for cloning the repository)
    *   Access to the APIs of AssemblyAI, Google Vertex AI, Cerebras, and OpenAI (if applicable). You will need to sign up and obtain API keys.
2.  **Clone the Repository:**
    
        git clone <your_repository_url>
        cd <your_repository_directory>
        
    
3.  **Create and Activate a Virtual Environment (Recommended):**
    
        python -m venv .venv
        # On Windows:
        .venv\Scripts\activate
        # On Linux/macOS:
        source .venv/bin/activate
        
    
4.  **Install Dependencies:**
    
        pip install -r requirements.txt  # (Create a requirements.txt file or install packages individually)
        # or install packages individually if there's no requirements.txt
        # pip install assemblyai google-cloud-aiplatform cerebras-sdk openai  # (Add only the needed libraries if you don't have requirements.txt)
        
    
5.  **Configure Environment Variables:**
    *   Create a `.env` file (or similar mechanism, depending on your preference) in the project root.
    *   Add your API keys:
        
            ASSEMBLYAI_API_KEY=<your_assemblyai_api_key>
            VERTEXAI_PROJECT_ID=<your_vertexai_project_id>
            VERTEXAI_LOCATION=<your_vertexai_location>  # e.g., us-central1
            CEREBRAS_API_KEY=<your_cerebras_api_key>   # if using Cerebras
            OPENAI_API_KEY=<your_openai_api_key>    # if using OpenAI
            
        
6.  **Running the Program:**
    *   Run the `runner.py` file. You'll likely need to provide a meeting audio input, potentially via a microphone or a pre-recorded audio file and specify parameters like LLM configurations. The specific implementation will require additional steps to access the input and trigger the relevant functions, which would be described in a longer Readme.

Challenges & Limitations
------------------------

*   **Speaker Diarization:** Lacks speaker identification within the transcription, leading to context limitations.
*   **LLM Costs and Rate Limits:** The cost and request limitations of LLMs require careful management and optimization strategies.
*   **Turn Length and Interruptions:** The program is less effective with frequent interruptions due to the lack of speaker diarization and the need to analyze conversational turns.

Future Development & Planned Improvements
-----------------------------------------

*   **Enhance User Interaction:** Integrate functionality to respond to questions and prompts the user to speak.
*   **Question Detection:** Implement a mechanism to identify questions directed toward the user within the conversation.
*   **Retrieval-Augmented Generation (RAG):** Incorporate a vector search or a personal knowledge base to provide relevant context for generating thoughts and responses. This allows the program to leverage a user's existing knowledge base.
*   **Complete Pipeline:** Build a complete pipeline: speech-to-text -> RAG -> LLM inferencing -> Response generation.
*   **Refactor Code:** Optimize existing code.

License
-------

None yet
