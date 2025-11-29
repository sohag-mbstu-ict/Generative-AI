from llama_index.llms.google_genai import GoogleGenAI
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os

venv_path = Path('/media/mtl/Volume F/PROJECTS/projects/.venv')  # or wherever your .venv is located
load_dotenv(dotenv_path=venv_path)
# Now access the API key
google_api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", google_api_key)


class GoogleGenAI_Chat_Model:
    def __init__(self,google_api_key):
        self.google_api_key = google_api_key

    def get_llm(self):
        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key = self.google_api_key
        )
        return llm

    def Basic_Usage_of_Chat_LLM(self):
        llm = self.get_llm()
        resp = llm.complete("Who is Paul Graham?")
        print(resp)

    async def Async_Usage(self):
        llm = self.get_llm()
        resp = await llm.astream_complete("Who is Paul Graham?")
        async for r in resp:
            print(r.delta, end="")
        print()


# --------------------------- Chat Model -----------------------------
gen_ai_obj = GoogleGenAI_Chat_Model(google_api_key)
# gen_ai_obj.Basic_Usage_of_Chat_LLM()
# ✅ RUN ASYNC METHOD PROPERLY
# asyncio.run(gen_ai_obj.Async_Usage())



from llama_index.llms.google_genai import GoogleGenAI
import google.genai.types as types
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from PIL import Image
from IPython.display import display

class Image_Generation_with_GoogleGenAI:
    def __init__(self):
        pass

    def Image_Generation_LLM(self):
        config = types.GenerateContentConfig(
        temperature=0.1, response_modalities=["Text", "Image"])

        llm = GoogleGenAI(
            model="models/gemini-2.0-flash-exp", generation_config=config)
        return llm
    
    def get_response(self):
        llm = self.Image_Generation_LLM()
        messages = [
        ChatMessage(role="user", content="Please generate an image of a cute dog")]
        resp = llm.chat(messages)
        return resp
    
    def get_generated_image(self, resp):
        for block in resp.message.blocks:
            if isinstance(block, ImageBlock):
                image = Image.open(block.resolve_image())
                image.show()  # ✅ Use this for scripts
                image.save("cute_dog.png")  # Optional: save it
            elif isinstance(block, TextBlock):
                print(block.text)


# --------------------------- Image generation Model -----------------------------

img_gen_obj = Image_Generation_with_GoogleGenAI()
resp = img_gen_obj.get_response()
img_gen_obj.get_generated_image(resp)
