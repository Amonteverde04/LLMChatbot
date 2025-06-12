from langchain.chat_models import init_chat_model
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import uuid
import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Init model.
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Prompt template set up.
prompt_template = ChatPromptTemplate.from_messages(
   [
        (
            "system",
            "You are a mysterious being. Answer all questions and statements in riddle with the following emotion: {emotion}. All riddles must directly reference message history or the users input."
        ),
        MessagesPlaceholder(variable_name="messages"),
   ]
)

emotion_template = ChatPromptTemplate.from_template(
    """
        Extract the desired information from the following message.
        Only extract the properties mentioned in the 'Emotion' function.

        Passage:
        {input}
    """
)

# Emotions for gemini to choose from.
available_emotions = [
    "Lonely",
    "Hurt",
    "Disappointed",
    "Caring",
    "Grateful",
    "Excited",
    "Respected",
    "Valued",
    "Accepted",
    "Brave",
    "Hopeful",
    "Powerful",
    "Creative",
    "Curious",
    "Affectionate",
    "Guilty",
    "Excluded",
    "Ashamed",
    "Annoyed",
    "Jealous",
    "Bored",
    "Overwhelmed",
    "Powerless",
    "Anxious",
]

# State class used for maintaining state.
class State(TypedDict):
   messages: Annotated[Sequence[BaseMessage], add_messages]

# Emotion class used for storing extracted emotions from a human message.
class Emotion(BaseModel):
    sentiment: str = Field(
        description="The detected sentiment of the text.",
        enum = available_emotions
    )

# Array representing chat history.
message_history = []

# Graph definition
workflow = StateGraph(state_schema=State)

# Model call definition
def call_model(state: State):
    # Get percieved emotion.
    structured_llm = model.with_structured_output(Emotion)
    emotion_prompt = emotion_template.invoke(
        {"input": state["messages"]}
    )
    emotion_response = structured_llm.invoke(emotion_prompt)

    # Create response.
    prompt = prompt_template.invoke(
       {"messages": state["messages"], "emotion": emotion_response.sentiment}
    )
    response = model.invoke(prompt)
    return {"messages": response}

# Define a node in the graph.
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Config to manage history with particular user.
# Useful to maintain different conversations from different users.
# Applied here for completeness.
config = {"configurable": {"thread_id": uuid.uuid1()}}

# Super Loop
query = ""
while (query.lower() != "bye"):
    query = input("Type your message to Gemini: ")
    print("\n")
    
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    print("\n")