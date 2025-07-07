import datetime
import logging
import os

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient, # Import StreamingClient for type hinting
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

from cerebras_summarizer import CerebrasSummarizer
from trend_detection import TrendDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionManager:
    """
    Manages the AssemblyAI transcription process and interactions with Cerebras for summarization.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.aai_client: StreamingClient | None = None 
        self.summarizer = CerebrasSummarizer()
        self.running_transcript = ""
        self.last_turn_transcript = ""
        self.summary_threshold_main_topics = 300 
        self.write_transcript_threshold = 2000 

        self.recent_transcript_char_lookback = 1000
        self.get_thought_char_lookback = 200

        self.running_main_topics: str | None = None  
        self.running_main_topics_list: list[str] = []  
        
        self.last_summary_point: int = 0 
        self.last_transcript_write_point: int = 0 

        self.recent_topics: str | None = None
        self.recent_topics_list: list[str] = [] 
        self.convo_direction: str | None = None 
        self.convo_direction_list: list[str] = [] 

        self.trend_detection: TrendDetection | None = None  

        self.turn_counter: int = 1
        self.start_time: datetime.datetime | None = None 
        self.end_time: datetime.datetime | None = None  
        self.last_turn_end_time: datetime.datetime | None = None

        self.time_since_last_thought: datetime.timedelta = datetime.timedelta(0)
        self.time_since_last_thought_threshold: datetime.timedelta = datetime.timedelta(seconds=5)
        self.last_transcript_write_point = 0

        self.thought_counter: int = 0

        self.recent_transcript: str = ""

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.transcript_filename = f"transcript_{timestamp_str}.txt"

    def initialize_clients(self):
        """Initializes the AssemblyAI client."""
        self.aai_client = StreamingClient(
            StreamingClientOptions(
                api_key = os.getenv("ASSEMBLYAI_API_KEY"),
                api_host="streaming.assemblyai.com",
            )
        )
        logger.info("AssemblyAI client initialized.")


    def on_begin(self, sdk_client: StreamingClient, event: BeginEvent):
        """Handles the beginning of a transcription session."""
        #logger.info(f"Session started via SDK client {sdk_client}: {event.id}")
        self.start_time = datetime.datetime.now()

    def update_last_turn_end_time(self):
        """Updates the last_turn_end_time if it is None, and returns the end_of_turn_timestamp and turn_duration."""
        end_of_turn_timestamp = datetime.datetime.now()
        if self.last_turn_end_time is None:
            self.last_turn_end_time = datetime.datetime.now()
        turn_duration = end_of_turn_timestamp - self.last_turn_end_time
        self.last_turn_end_time = end_of_turn_timestamp

        return end_of_turn_timestamp, turn_duration

    def on_turn(self, sdk_client: StreamingClient, event: TurnEvent):
        """Handles each turn of the transcription, including summarization."""
        #logger.info(f"Turn: {event.transcript} ({event.end_of_turn})")
        print(f"Turn: {event.transcript} ({event.end_of_turn})")

        can_think = False

        if event.end_of_turn:
            end_of_turn_timestamp, turn_duration = self.update_last_turn_end_time()
            if turn_duration + self.time_since_last_thought > self.time_since_last_thought_threshold:
                can_think = True
                self.time_since_last_thought = datetime.timedelta(0)
            else:
                can_think = False
                self.time_since_last_thought += turn_duration

            print(f"can_think: {can_think}, time_since_last_thought: {self.time_since_last_thought}, turn_duration: {turn_duration}")
                
            self.last_turn_transcript = event.transcript
            self.running_transcript += event.transcript + "\n"
            full_char_count = len(self.running_transcript)
            turn_char_count = len(event.transcript)
            char_stats = f'{full_char_count-turn_char_count} + {turn_char_count} = {full_char_count}'
            print(f'{end_of_turn_timestamp}: ({self.turn_counter}) [{char_stats}] {turn_duration}')
            
            #logger.debug(f"Running transcript: {self.running_transcript.strip()}")

            print(f'len(self.running_transcript): {len(self.running_transcript)}')
            print(f'current_length - self.last_summary_point: {len(self.running_transcript) - self.last_summary_point}')

            current_length = len(self.running_transcript)
            summary_progress = current_length - self.last_summary_point
            should_summarize = summary_progress >= self.summary_threshold_main_topics

            write_transcript_progress = current_length - self.last_transcript_write_point
            should_write_transcript = write_transcript_progress >= self.write_transcript_threshold

            if should_write_transcript:
                transcript_segment_to_write = self.running_transcript[self.last_transcript_write_point:current_length]

                #print(f'writing segment to file: {transcript_segment_to_write}')

                with open(self.transcript_filename, "a") as f:
                    f.write(transcript_segment_to_write + "\n")
                self.last_transcript_write_point = current_length
            
            #logger.debug(f"Summary progress: {summary_progress}, Threshold: {self.summary_threshold_main_topics}, Should summarize: {should_summarize}")
            if should_summarize and 1 == 1:
                try:
                    print("running llms...")
                    self.running_main_topics = self.summarizer.summarize_with_vertexai(self.running_transcript)
                    self.running_main_topics_list.append(self.running_main_topics)  # Store main topics
                    #logger.info(f"Generated main topics summary: {self.running_main_topics}")
                    self.last_summary_point = current_length  # Update summary point

                    #print('')

                    self.recent_transcript = self.running_transcript[-self.recent_transcript_char_lookback:]
                    self.recent_topics = self.summarizer.summarize_with_vertexai(self.recent_transcript)  # Summarize the recent part
                    self.recent_topics_list.append(self.recent_topics) 
                    #print(f'recent topics: {self.recent_topics}')

                    self.trend_detection = self.summarizer.detect_trends(
                        self.running_main_topics,
                        self.recent_topics)  # Detect trends based on summaries
                    
                    msg = f"""
                        latest transcript segment:
                        {self.recent_transcript}

                        emerging:
                        {self.trend_detection.emerging_topics}

                        fading:
                        {self.trend_detection.fading_topics}

                        shifting:
                        {self.trend_detection.shifting_emphasis}
                    """
                    print(msg)

                except Exception as e:
                    logger.error(f"Error during VertexAI main topics summarization: {e}")
                    self.running_main_topics = "(Main Topics Summary Error)"

            if can_think and self.trend_detection is not None:
                try:
                    full_summary = []

                    for summary_chunk in self.summarizer.get_cerebras_thought(
                        self.running_transcript[-self.get_thought_char_lookback:], self.trend_detection):
                        full_summary.append(summary_chunk)
                    print() 
                    print(f"{''.join(full_summary)}")

                except Exception as e:
                    logger.error(f"Error during summary generation: {e}")

            self.turn_counter += 1
        if event.end_of_turn and not event.turn_is_formatted:
            params = StreamingSessionParameters(
                format_turns=False,
            )

            sdk_client.set_params(params)


    def on_terminated(self, sdk_client: StreamingClient, event: TerminationEvent):
        """Handles the termination of a transcription session."""
        logger.info(
            f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
        )


    def on_error(self, sdk_client: StreamingClient, error: StreamingError):
        """Handles errors during transcription."""
        logger.error(f"Error occurred: {error}")

    def start_streaming(self):
        """Starts the AssemblyAI streaming transcription process."""
        if not self.aai_client:
            raise ValueError("AssemblyAI client not initialized. Call initialize_clients() first.")

        self.aai_client.on(StreamingEvents.Begin, self.on_begin)
        self.aai_client.on(StreamingEvents.Turn, self.on_turn)
        self.aai_client.on(StreamingEvents.Termination, self.on_terminated)
        self.aai_client.on(StreamingEvents.Error, self.on_error)

        logger.info("Connecting to AssemblyAI streaming service...")
        self.aai_client.connect(
            StreamingParameters(
                sample_rate=self.sample_rate,
                format_turns=False, # Initial setting for turns
            )
        )
        logger.info("Connected. Starting microphone stream...")

        try:
            self.aai_client.stream(
              aai.extras.MicrophoneStream(sample_rate=self.sample_rate)
            )
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
        finally:
            logger.info("Disconnecting from AssemblyAI streaming service.")
            self.aai_client.disconnect(terminate=True)
            logger.info("Disconnected.")


def main():
    """Main function to run the transcription and summarization."""
    logger.info("Initializing Transcription Manager...")

    transcription_manager = TranscriptionManager()
    transcription_manager.initialize_clients()

    logger.info("Starting streaming...")
    transcription_manager.start_streaming()
    logger.info("Streaming finished.")


if __name__ == "__main__":
    main()