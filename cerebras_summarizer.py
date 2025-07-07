import logging
from typing import Type, Generator # Keep Type for StreamingClient type hint
import os

from cerebras.cloud.sdk import Cerebras

from vertexai.generative_models import (
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Part,
    SafetySetting,
)
import vertexai

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List

from trend_detection import TrendDetection
from conversational_markers import ConversationalMarkers

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)



class CerebrasSummarizer:
    """
    Handles the summarization of transcripts using the Cerebras API.
    """

    def __init__(self: str):
        self.cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f'./ai-text-401001-265f1d68ac1d.json'
        vertexai.init(project="ai-text-401001", location="us-central1")
        parameters = {
            "max_output_tokens": 2048,
            "temperature": 0.2
        }

        self.summarize_prompt = "in 100 words or less, what are the main points of the following text"

    def detect_trends(self, running_summary: str, recent_summary: str) -> str:
        prompt = f"""
        Analyze the current focus of the conversation based on the following information:

        CURRENT OVERALL SUMMARY OF MAIN TOPICS (reflecting the entire conversation up to this point):
        {running_summary}

        SUMMARY OF THE MOST RECENT DISCUSSION SEGMENT:
        {recent_summary}

        Based on this, please identify:
        1. KEY POINTS IN RECENT DISCUSSION: What are the most distinct points, subjects, or themes highlighted in the 'SUMMARY OF THE MOST RECENT DISCUSSION SEGMENT'?
        2. RELATIONSHIP TO OVERALL SUMMARY: How do these 'KEY POINTS IN RECENT DISCUSSION' relate to the 'CURRENT OVERALL SUMMARY OF MAIN TOPICS'?
            - Do they introduce a more specific aspect of an overall topic?
            - Do they appear to be a primary focus within the broader conversation at this moment?
            - Do they highlight a particular development, detail, or conclusion related to an overall topic?
        3. CURRENT CONVERSATIONAL EMPHASIS: Considering both summaries, what appears to be the primary emphasis or the direction the conversation is taking right now based *only* on this current information?

        Provide a concise analysis for each category in 100 words or less.
        (Note: True "emerging topics" from a historical perspective, "fading topics," or "shifts in emphasis over time" cannot be reliably determined without comparing to a previous overall summary.)
        """
        m = []
        m.append({
            'role': 'system',
            'content': prompt
        })

        chat_completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Or another supported model
            messages=m,
            response_format=TrendDetection,
        )

        #re = json.dumps(chat_completion.dict(), indent=2)
        message = chat_completion.choices[0].message
        if (message.parsed):
            return message.parsed
        else:
            return ""


    def get_conversational_markers(self, recent_transcript) -> str:
        prompt = f"""
        Analyze the following conversation transcript segment:
        ---
        {recent_transcript}
        ---

        Based ONLY on the text provided above, please identify and list the following key conversational markers.
        For each category, state the findings. If no findings for a category, state 'None for [Category Name]'.

        1. QUESTIONS & ANSWERS:
           - Identify any distinct questions asked.
           - For each question, if an answer is explicitly provided within this segment, note the answer.
           Example:
             - Q: What is the deadline? A: The deadline is Friday.
             - Q: Who is responsible for the report? A: (No answer in this segment)

        3. ACTION ITEMS:
           - Identify any tasks assigned to someone or a commitment to do something.
           Example:
             - John will draft the proposal.
             - Action: Send out the meeting minutes by EOD.

        Please format your response clearly, addressing each numbered category.
        """
        m = []
        m.append({
            'role': 'system',
            'content': prompt
        })

        chat_completion = self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Or another supported model
            messages=m,
            response_format=ConversationalMarkers,
        )

        #re = json.dumps(chat_completion.dict(), indent=2)
        message = chat_completion.choices[0].message
        if (message.parsed):
            return message.parsed
        else:
            return ""

    def prompt_vertexai(self, prompt: str) -> str:

        # print('')
        # print('Summarizing with VertexAI...')
        # print('')

        chat_history = []
        chat_history.append({
            'role': 'user',
            'parts': [ {'text': prompt} ]
        })
        model = GenerativeModel("gemini-2.0-flash-lite")
        safety_config = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        re_summary = model.generate_content(chat_history, safety_settings=safety_config).text

        #print(re_summary)

        return re_summary
    
    def get_cerebras_thought(self, turn_transcript: str, trend_detection: TrendDetection) -> Generator[str, None, None]:

        trend_direction_str = f"""
emerging:
{trend_detection.emerging_topics}

fading:
{trend_detection.fading_topics}

shifting:
{trend_detection.shifting_emphasis}
                    """

#         thought_prompt = f"""in 10 words or less, give me a concise thought 
# based on the following [[conversation segment]] and [[conversation trend analysis]]:

# [[conversation segment]]:
# {turn_transcript}

# [[conversation trend analysis]]:
# {trend_direction_str}
# The thought should be a single sentence that captures the essence of the conversation segment and how it relates to the overall trend.
# The thought should be concise, clear, and insightful, providing a deeper understanding of the conversation segment in the context of the trend analysis.
# Use 10 words or less.
#         """

        thought_prompt = f"""in 10 words or less, give me a concise thought trigger
based on the following [[conversation segment]] and [[conversation trend analysis]]:

[[conversation segment]]:
{turn_transcript}

[[conversation trend analysis]]:
{trend_direction_str}
The trigger should be a cluster of words that captures the essence of the conversation segment 
and how it relates to the overall trend, and encourages further exploration or discussion.
The thought should be concise, clear, and insightful, providing a 
deeper understanding of the conversation segment in the context of the trend.
Use 10 words or less. Give me the thought trigger only and nothing else.
        """

        # print('Thought Prompt:')    
        # print(thought_prompt)

        try:
            stream = self.cerebras_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{thought_prompt}"
                    }
                ],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.2,
                top_p=1
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            yield " (Summary Error)"   

    def summarize_with_cerebras(self, transcript: str) -> Generator[str, None, None]:
        """
        Summarizes the given transcript using the Cerebras API.

        Yields:
            str:  Content chunks from the Cerebras API response (stream).
        """

        # print('')
        # print('Summarizing with Cerebras...')
        # print('')

        try:
            stream = self.cerebras_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{self.summarize_prompt}: {transcript}"
                    }
                ],
                model="llama-4-scout-17b-16e-instruct",
                stream=True,
                max_completion_tokens=2048,
                temperature=0.2,
                top_p=1
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            yield " (Summary Error)"

    def summarize_with_vertexai(self, transcript: str) -> str:
        prompt = f'{self.summarize_prompt}: {transcript}'
        return self.prompt_vertexai(prompt)
    
    def summarize_recent_with_vertexai(
            self, 
            running_main_topics: str,
            recent_summary: str) -> str:
        
        prompt = f'''given the following overarching summary of a conversation [[convo]],
and the summary of the recent conversation [[recent_summary]],
give me 5 bullet points, each 9 words or less, that summarize the recent conversation
and the conversation overall (considering the overaching summary and how the recent
conversation fits into it [i.e. how the conversation is moving and what
it is about now]) in order of most recent to overarching:

[[convo]]: {running_main_topics}
[[recent_summary]]: {recent_summary}
        '''

        return self.prompt_vertexai(prompt)
    
    def identify_trend_direction(
            self, 
            running_main_topics: str,
            recent_summary: str) -> str:
        
        prompt = f'''given the following overarching summary of a conversation [[convo]],
and the summary of the recent conversation [[recent_summary]],
give me 5 bullet points, each 9 words or less, that summarize the recent conversation
and the conversation overall (considering the overaching summary and how the recent
conversation fits into it [i.e. how the conversation is moving and what
it is about now]) in order of most recent to overarching:

[[convo]]: {running_main_topics}
[[recent_summary]]: {recent_summary}
        '''

        return self.prompt_vertexai(prompt)


    def summarize(self, transcript: str) -> Generator[str, None, None]:
        # Alternate between Cerebras and VertexAI to avoid rate limits
        if not hasattr(self, '_use_cerebras_next'):
            self._use_cerebras_next = True
        use_cerebras = self._use_cerebras_next
        self._use_cerebras_next = not self._use_cerebras_next

        if use_cerebras:
            # Use Cerebras streaming summarization (yields chunks)
            yield from self.summarize_with_cerebras(transcript)
        else:
            # Use VertexAI summarization (returns full string)
            try:
                summary = self.summarize_with_vertexai(transcript)
                yield summary
            except Exception as e:
                logger.error(f"Error during VertexAI summarization: {e}")
                yield " (Summary Error)"