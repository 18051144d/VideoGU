import base64

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class Scorer_1(BaseModel):
    PROCEDURE: str = Field(description='Return the maximum 3 potential procedure code (1-177), code 0 if it is not a procedure, as a list of int in EXACTLY the following format: [code1, code2, code3]. Seperate the code with comma ONLY.')
    EXPLANATION: str = Field(description="Provide a concise one-sentence explanation for your chosen answer.")

class Scorer_2(BaseModel):
    PROCEDURE: str = Field(description='Return the maximum potential procedure code (1-3), code 0 if it is not a procedure, as an int.')
    EXPLANATION: str = Field(description="Provide a concise one-sentence explanation for your chosen answer.")

class GeminiFlash:
    # Function to encode the local image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def __call__(self, image_path, prompt, round_id =1, **kwargs):
        image_data = self.encode_image(image_path)
        model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash')
        SLLM = model.with_structured_output(Scorer_1) if round_id == 1 else model.with_structured_output(Scorer_2)
        caption = SLLM.invoke([
                HumanMessage(content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ])

            ])
        return caption
