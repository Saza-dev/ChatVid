import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class FrameCaptioner:
    def __init__(self, model_name: str = os.getenv("CAP_MODEL")):
        """
        Initializes the Groq client.

        Args:
            model_name (str): The name of the vision model to use on the Groq platform.
        """
        self.model = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))


    def caption_image(self, image_path: str, prompt: str = "Describe this image concisely.") -> str:
        """
        Takes a local image path, sends it to the Groq API, and returns a caption.

        Args:
            image_path (str): The file path to the local image.
            prompt (str): The text prompt to guide the captioning.

        Returns:
            str: The generated caption from the model.
        """
        try:
            # 1. Read and Base64 encode the image
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 2. Make the API call to Groq
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                        ],
                    }
                ],
                model=self.model,
            )
            
            # 3. Return the content of the response
            return chat_completion.choices[0].message.content

        except FileNotFoundError:
            return f"[Error: Image file not found at {image_path}]"
        except Exception as e:
            return f"[Error: Failed to generate caption - {e}]"