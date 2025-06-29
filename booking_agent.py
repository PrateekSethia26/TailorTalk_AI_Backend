from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
from contextlib import asynccontextmanager

# Import your existing code
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
import getpass
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import CalendarToolkit
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_google_community.calendar.utils import build_resource_service
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


load_dotenv()
# Global variables
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing Calendar Assistant...")
    
    # Initialize your existing code
    memory = MemorySaver()
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    current_time = datetime.now().strftime("%I:%M %p")

    SYSTEM_PROMPT = f"""You are a helpful calendar assistant. Today's date is {current_date} and the current time is {current_time}. 

    When users ask about scheduling events, meetings, or calendar-related tasks, always consider this current date and time context. You can help users:

    - Important Thing when creating a meeting check first if there is already a meeting scheduled. if so ask the user if he wants to delete the meeting for schedule it on another time.
    - Create new calendar events
    - Check their calendar for availability
    - Update or modify existing events
    - Answer questions about their schedule

    Always be precise with dates and times, and ask for clarification if the user's request is ambiguous about timing."""

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    
    def getAccessToken():
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=8000)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    def getCalenderTools():
        credentials = getAccessToken()
        api_resource = build_resource_service(credentials=credentials)
        toolkit = CalendarToolkit(api_resource=api_resource)
        return toolkit.get_tools()

    # Set up Google AI API key
    if "GOOGLE_API_KEY" not in os.environ:
        # For production, you should set this as an environment variable
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY 

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    tools = getCalenderTools()
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        messages = state["messages"]
        has_system_message = any(msg.get("role") == "system" for msg in messages if hasattr(msg, 'get') or isinstance(msg, dict))
        
        if not has_system_message:
            system_message = {"role": "system", "content": SYSTEM_PROMPT}
            messages_with_system = [system_message] + messages
        else:
            messages_with_system = messages
        
        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": [response]}

    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    graph = graph_builder.compile(checkpointer=memory)
    
    # Store in global state
    app_state["graph"] = graph
    app_state["current_date"] = current_date
    app_state["current_time"] = current_time
    
    print(f"Calendar Assistant initialized. Today is {current_date}")
    
    yield
    
    # Shutdown
    print("Shutting down Calendar Assistant...")

app = FastAPI(title="Calendar Assistant API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    thread_id: str = "1"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

@app.get("/")
async def root():
    return {
        "message": "Calendar Assistant API", 
        "current_date": app_state.get("current_date", "Not initialized"),
        "current_time": app_state.get("current_time", "Not initialized")
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        if "graph" not in app_state:
            raise HTTPException(status_code=500, detail="Calendar Assistant not initialized")
        
        graph = app_state["graph"]
        config = {"configurable": {"thread_id": message.thread_id}}
        
        # Stream the graph updates
        events = graph.stream(
            {"messages": [{"role": "user", "content": message.message}]},
            config,
            stream_mode="values",
        )
        
        # Get the last response
        last_response = None
        for event in events:
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content'):
                    last_response = last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    last_response = last_message['content']
        
        if last_response is None:
            last_response = "I'm sorry, I couldn't process your request. Please try again."
        
        return ChatResponse(response=last_response, thread_id=message.thread_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "initialized": "graph" in app_state}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)