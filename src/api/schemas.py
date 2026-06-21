"""FastAPI request and response schemas."""

from time import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from router.schemas import ChatMessage, RouterResponse


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    user: str | None = None


class Usage(BaseModel):
    """OpenAI-compatible token usage payload."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    """OpenAI-compatible assistant message payload."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response with router metadata."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    router: RouterResponse

    @classmethod
    def from_router_response(cls, router_response: RouterResponse) -> "ChatCompletionResponse":
        """Build an OpenAI-compatible response from a router response.

        Args:
            router_response: Router response.

        Returns:
            OpenAI-compatible completion response.
        """
        return cls(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time()),
            model=router_response.model_used,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=router_response.content),
                ),
            ],
            usage=Usage(
                prompt_tokens=router_response.input_tokens,
                completion_tokens=router_response.output_tokens,
                total_tokens=router_response.input_tokens + router_response.output_tokens,
            ),
            router=router_response,
        )
