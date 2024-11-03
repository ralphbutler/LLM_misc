import chainlit as cl
import pandas as pd
import io

@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        for element in message.elements:
            # Handle CSV files
            if element.mime == 'text/csv':
                df = pd.read_csv(io.BytesIO(element.content))
                await cl.Message(f"CSV contents:\n{df.head()}").send()
            
            # Handle images
            elif element.mime.startswith('image/'):
                # Save the image
                with open(f"images/{element.name}", "wb") as f:
                    f.write(element.content)
                await cl.Message(f"Saved image: {element.name}").send()
            
            # Handle PDFs
            elif element.mime == 'application/pdf':
                # Process PDF file
                with open(f"pdfs/{element.name}", "wb") as f:
                    f.write(element.content)
                await cl.Message(f"Saved PDF: {element.name}").send()
            
            else:
                await cl.Message(f"Received file of type {element.mime}").send()
    else:
        await cl.Message("No file was attached").send()
