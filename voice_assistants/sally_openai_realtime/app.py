import os
import asyncio
from openai import AsyncOpenAI

import chainlit as cl
from uuid import uuid4
from chainlit.logger import logger

from realtime import RealtimeClient
from tools import tools

client = AsyncOpenAI()    

async def setup_openai_realtime():
    """Instantiate and configure the OpenAI Realtime Client"""
    openai_realtime = RealtimeClient(api_key=os.getenv("OPENAI_API_KEY"))
    cl.user_session.set("track_id", str(uuid4()))
    async def handle_conversation_updated(event):
        delta = event.get("delta")
        """Currently used to stream audio back to the client."""
        if delta and 'audio' in delta:
            audio = delta['audio']  # Int16Array, audio added
            await cl.context.emitter.send_audio_chunk(cl.OutputAudioChunk(mimeType="pcm16", data=audio, track=cl.user_session.get("track_id")))
            
    async def handle_item_completed(event):
        """Used to populate the chat context with transcription once an item is completed."""
        item = event.get("item")
        if item and item.get("role") == "assistant":
            transcript = item.get("formatted", {}).get("transcript", "")
            if transcript:
                await cl.Message(content=transcript, author="Assistant").send()
    
    async def handle_conversation_interrupt(event):
        """Used to cancel the client previous audio playback."""
        cl.user_session.set("track_id", str(uuid4()))
        await cl.context.emitter.send_audio_interrupt()
        
    async def handle_error(event):
        logger.error(event)
        
    
    openai_realtime.on('conversation.updated', handle_conversation_updated)
    openai_realtime.on('conversation.item.completed', handle_item_completed)
    openai_realtime.on('conversation.interrupted', handle_conversation_interrupt)
    openai_realtime.on('error', handle_error)

    cl.user_session.set("openai_realtime", openai_realtime)
    coros = [openai_realtime.add_tool(tool_def, tool_handler) for tool_def, tool_handler in tools]
    await asyncio.gather(*coros)


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome to the Chainlit x OpenAI realtime example. Press `P` to talk!"
    ).send()
    await setup_openai_realtime()

@cl.on_message
async def on_message(message: cl.Message):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        ## RMB added this if-block
        if message.elements:
            for element in message.elements:
                if element.mime.startswith('image/'):
                    print("DBGIMG",element.name)
                    cl.user_session.set("image_filename", element.name)
                    # await cl.Message(f"GOT image: {element.name}").send()
                    await openai_realtime.send_user_message_content(
                            [
                                { "type": 'input_text', "text": message.content },
                            ]
                    )
        else:
            cl.user_session.set("image_filename", "NONE")
            await openai_realtime.send_user_message_content([{ "type": 'input_text', "text": message.content }])
    else:
        await cl.Message(content="Please activate voice mode before sending messages!").send()

@cl.on_audio_start
async def on_audio_start():
    try:
        openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
        await openai_realtime.connect()
        logger.info("Connected to OpenAI realtime")
        # TODO: might want to recreate items to restore context
        # openai_realtime.create_conversation_item(item)
        return True
    except Exception as e:
        await cl.ErrorMessage(content=f"Failed to connect to OpenAI realtime: {e}").send()
        return False

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")            
    if openai_realtime.is_connected():
        await openai_realtime.append_input_audio(chunk.data)
    else:
        logger.info("RealtimeClient is not connected")

@cl.on_audio_end
@cl.on_chat_end
@cl.on_stop
async def on_end():
    openai_realtime: RealtimeClient = cl.user_session.get("openai_realtime")
    if openai_realtime and openai_realtime.is_connected():
        await openai_realtime.disconnect()
