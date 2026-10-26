"""One local Ollama boundary with bounded output, timeouts, and typed JSON."""

import json
import logging
from urllib.request import urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.settings = settings

    def status(self) -> tuple[bool, str]:
        try:
            with urlopen(f"{self.settings.ollama_base_url}/api/tags", timeout=3) as response:
                models = json.load(response).get("models", [])
            names = {model["name"] for model in models}
            requested = self.settings.ollama_model
            if requested not in names and f"{requested}:latest" not in names:
                return False, f"Model missing. Run: ollama pull {requested}"
            return True, f"{requested} is ready"
        except Exception:
            return False, "Ollama is unavailable. Install Ollama, then run ollama serve."

    def generate(self, system: str, payload: dict, schema: type[BaseModel]) -> BaseModel:
        output_schema = schema.model_json_schema()
        # Ollama's format constrains decoding; it does not explain the response
        # contract to the model. Include the very same schema in the instructions.
        instructions = system + "\nReturn only a JSON object matching this response schema:\n" + json.dumps(output_schema)
        model = ChatOllama(
            model=self.settings.ollama_model, base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature, num_ctx=self.settings.llm_context_tokens,
            num_predict=1200, format=output_schema,
            client_kwargs={"timeout": self.settings.llm_timeout_seconds},
        )
        try:
            response = model.invoke([
                SystemMessage(content=instructions),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ])
            return schema.model_validate_json(response.content)
        except ValidationError as exc:
            log.exception("Invalid structured model output")
            raise LLMError("The local model returned an incomplete response. Try a shorter question or a larger Ollama model.") from exc
        except Exception as exc:
            log.exception("Ollama inference failed")
            raise LLMError(
                f"Could not generate a response. Check that Ollama is running and run: ollama pull {self.settings.ollama_model}. "
                "If inference timed out, increase LLM_TIMEOUT_SECONDS in .env."
            ) from exc
