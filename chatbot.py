from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import SystemMessage, trim_messages
import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

class State(TypedDict):
   messages: Annotated[Sequence[BaseMessage], add_messages]
   language: str

trimmer = trim_messages(
    max_tokens = 65,
    strategy = "last",
    token_counter = model,
    include_system = True,
    allow_partial = False,
    start_on = "human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

prompt_template = ChatPromptTemplate.from_messages(
   [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}."
        ),
        MessagesPlaceholder(variable_name="messages"),
   ]
)

# Define a new graph.
workflow = StateGraph(state_schema=State)

# Define the function that calls the model
def call_model(state: State):
   trimmed_messages = trimmer.invoke(state["messages"])
   prompt = prompt_template.invoke(
      {"messages": trimmed_messages, "language": state["language"]}
   )
   response = model.invoke(prompt)
   return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = messages + [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode = "messages",
):
    if isinstance(chunk, AIMessage):
       print(chunk.content, end = "|")