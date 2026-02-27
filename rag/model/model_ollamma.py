from rag.model.model_Provider import ModelProvider,ModelSpec
from typing import Optional, Dict,List,Union
import json 
import urllib.request
class OllamaModel(ModelProvider):
    def __init__(self,spec:ModelSpec,url:Optional[str]="http://localhost:11434/api/chat",stream:Optional[bool]=False):

        self.model_name=spec.model_name
        self.url=url
        self.stream=stream

    def get_response(self, message: Union[str, List[Dict[str, str]]]) -> str:

        if isinstance(message,str):
            messages = [{"role": "user", "content": message}]
        elif isinstance(message,Dict) and "system" in message and "user" in message:
            messages = [
            {"role": "system", "content": message["system"]},
            {"role": "user", "content": message["user"]},
                ]
        else:
            messages=message

        payload={
            'model':self.model_name,
            'messages':messages,
            "stream":self.stream
        }

        req=urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req) as resp:
            out= json.loads(resp.read().decode("utf-8"))
        return out["message"]["content"]
    


