import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel,RunConfig,Agent,Runner
import chainlit as cl

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI API KE is not set...")


provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

config= RunConfig(
    model= model,
    model_provider=provider,
    tracing_disabled=True
)

agent= Agent(
    name="Assistant",
    instructions="You are a medical Assistant.",
    model=model
)


result = Runner.run_sync(
    agent, 
    "What is Cardiac Arrest?",
    run_config=config
)

print(result.final_output)


@cl.on_chat_start
async def welcome():
    await cl.Message(
        author="AI Assistant",
        content="Welcome"
    ).send()
    
    
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.strip()
    
    try:
        loading_msg = cl.Message(author="🤖", content="**Generating response...**")
        await loading_msg.send()
      
        result = await Runner.run(agent, input=user_input, run_config=config)

        loading_msg.content = f"### Agent's Response\n\n{result.final_output.strip()}"
        await loading_msg.update()

    except Exception as e:
        print("Error:", e)
        await cl.Message(
            author=" Error",
            content="Oops! Something went wrong while processing your request. Please try again."
        ).send()