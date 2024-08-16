import sys, os, json

import PIL.Image

from pydantic import BaseModel
from typing import List

import google.generativeai as genai
import typing_extensions as typing


img = PIL.Image.open('flight_schedule.png')

class FlightInfo(typing.TypedDict):
    flight_time: str
    flight_destination: str

# Using response_mime_type with response_schema requires a Gemini 1.5 Pro model
model = genai.GenerativeModel(
    "gemini-1.5-pro",
    generation_config={"response_mime_type": "application/json",
                       "response_schema": list[FlightInfo]})

prompt = "List out the flight times and destinations"

response = model.generate_content([prompt,img])

print(response.text)
print("-"*50)

for flight in json.loads(response.text):
    # print("FLIGHT:",flight)
    print(flight['flight_time'], flight['flight_destination'])
print("-"*50)


class FlightInfo(BaseModel):
    flight_time: str
    flight_destination: str

json_schema = FlightInfo.schema_json(indent=2)
print(json_schema)
print("-"*50)

model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction=f"""
        You are a helpful assistant that scans for flight times and destinations.
        Use this JSON schema:
            FlightInfo = {json_schema}
        Return a `list[FlightInfo]`

    """,
    generation_config={"response_mime_type": "application/json", }
)

prompt = "List out the flight times and destinations"

response = model.generate_content([prompt,img])
print(response)
print("-"*50)

for flight in json.loads(response.text):
    print(flight['flight_time'], flight['flight_destination'])
print("-"*50)

