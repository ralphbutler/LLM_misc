
import sys, os, json, textwrap
from openai import OpenAI

class FauxGrimoire:
    def __init__(self,model="gpt-4-1106-preview"):
        self.model = model
        self.client = OpenAI()
        self.prev_code = "none"
        self.sys_msg = textwrap.dedent("""
            You are an expert at generating python code.
            You should formulate a step-by-step plan each time you generate code.
            Always return your answer in JSON format, like this:
                {
                    "step_by_step_plan": <plan>,
                    "code": <code>
                }
        """)

    def generate_new_code(self,user_input):
        user_msg = textwrap.dedent(f"""
            PREVIOUS CODE:
            {self.prev_code}
            USER INPUT:
            {user_input}
        """)
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},   # this matters
            messages = [
                { "role": "system", "content": self.sys_msg},
                { "role": "user",   "content": user_msg},
            ]
        )
        jresponse = json.loads(response.choices[0].message.content)
        self.prev_code = jresponse["code"]
        return response

def main():
    faux_grimoire = FauxGrimoire()
    while True:
        user_input = input("Anonymous: ")
        print("INPUT:",user_input)
        response = faux_grimoire.generate_new_code(user_input)
        jresponse = json.loads(response.choices[0].message.content)
        print("\nStep-by-Step Plan:")
        for line in jresponse["step_by_step_plan"]:
            print(f"    {line}")
        print("\nCode:")
        print(jresponse["code"])
        print("-"*50,"\n")

main()
