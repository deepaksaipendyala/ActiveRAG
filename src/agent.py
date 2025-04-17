import re
import time
import asyncio
import types
from typing import Union

# Import the OllamaClient from your implementation
class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", default_model="llama3.2"):
        self.base_url = base_url
        self.default_model = default_model

    def chat_completion_create(self, messages, model=None, temperature=0.2, n=1):
        # Build the prompt from the conversation history
        prompt_text = ""
        for m in messages:
            prompt_text += f"{m['role'].capitalize()}: {m['content']}\n"
        prompt_text += "Assistant:"

        used_model = model if model else self.default_model
        payload = {
            "prompt": prompt_text,
            "model": used_model,
            "temperature": temperature,
            "stream": False
        }

        import requests
        resp = requests.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        text = data.get("response", "") or data.get("generated_text", "")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text
                    }
                }
            ]
        }

# Instantiate the Ollama client
MODEL = "llama3.2"
client = OllamaClient(default_model=MODEL)
DEFAULT_PROMPT = ""

class Agent:
    def __init__(self, template, model=MODEL, key_map: Union[dict, None] = None) -> None:
        self.message = []

        if isinstance(template, str):
            self.TEMPLATE = template
            self.template_list = None
        elif isinstance(template, list) and len(template) > 1:
            self.TEMPLATE = template[0]
            self.template_list = template
        else:
            self.TEMPLATE = ""
            self.template_list = None

        self.key_map = key_map
        self.model = model

        self.func_dic = {}
        self.func_dic['default'] = self.get_output
        self.func_dic['padding_template'] = self.padding_template

    def send_message(self):
        assert len(self.message) != 0 and self.message[-1]['role'] != 'assistant', \
            'ERROR in message format: last message must be from user.'
        try:
            ans = client.chat_completion_create(
                messages=self.message,
                model=self.model,
                temperature=0.2,
                n=1
            )
            self.parse_message(ans)
            return ans
        except Exception as e:
            print("Error in send_message:", e)
            time.sleep(5)
            ans = client.chat_completion_create(
                messages=self.message,
                model=self.model,
                temperature=0.2,
                n=1
            )
            self.parse_message(ans)
            return ans

    async def send_message_async(self, session):
        return self.send_message()

    def padding_template(self, input_dict):
        input_dict = self.key_mapping(input_dict)
        assert self._check_format(input_dict.keys()), \
            f"Input lacks necessary keys for template: {self.TEMPLATE}"
        msg = self.TEMPLATE.format(**input_dict)
        self.message.append({'role': 'user', 'content': msg})

    def key_mapping(self, input_dict):
        if self.key_map is not None:
            new_input = {}
            for key, val in input_dict.items():
                if key in self.key_map:
                    new_input[self.key_map[key]] = val
                else:
                    new_input[key] = val
            input_dict = new_input
        return input_dict

    def _check_format(self, key_list):
        placeholders = re.findall(r'\{([^}]+)\}', self.TEMPLATE)
        for key in placeholders:
            if key not in key_list:
                return False
        return True

    def get_output(self) -> str:
        assert len(self.message) != 0 and self.message[-1]['role'] == 'assistant', \
            "No assistant response yet."
        return self.message[-1]['content']

    def get_final_answer(self):
        """
        Extract the final answer from assistant's last message.
        """
        response = self.get_output()
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip()

    def parse_message(self, completion):
        content = completion["choices"][0]["message"]["content"]
        role = completion["choices"][0]["message"]["role"]
        record = {'role': role, 'content': content}
        self.message.append(record)
        return record

    def parse_message_json(self, completion):
        return self.parse_message(completion)

    def regist_fn(self, func, name):
        setattr(self, name, types.MethodType(func, self))
        self.func_dic[name] = getattr(self, name)
