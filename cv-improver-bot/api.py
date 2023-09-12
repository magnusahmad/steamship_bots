from typing import List, Optional, Type
import logging
import json
from pydantic import Field
from pydantic.main import BaseModel
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI, OpenAI
from steamship.agents.mixins.transports.slack import (
    SlackTransport,
    SlackTransportConfig,
)
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.mixins.transports.telegram import (
    TelegramTransport,
    TelegramTransportConfig,
)
from steamship.agents.schema import Tool, ChatLLM, LLM
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import Config, post
from steamship.utils.kv_store import KeyValueStore
from steamship.utils.repl import AgentREPL
from steamship.agents.logging import AgentLogging
from steamship import Block, File, PluginInstance, Steamship, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import GenerationTag

# Save original __init__
original_init = ChatOpenAI.__init__

def new_init(self, client, model_name="gpt-4-0613", **kwargs):

    # Update kwargs
    kwargs["max_tokens"] = 1500
    
    # Call original __init__
    original_init(self, client, model_name, **kwargs) 

# Override __init__ on class
ChatOpenAI.__init__ = new_init

# class ChatOpenAI(ChatLLM, OpenAI):

#     def __init__(self, client, model_name="gpt-4-061", **kwargs):
        
#         # Update max_tokens
#         kwargs["max_tokens"] = 10000
        
#         super().__init__(client=client, model_name=model_name, **kwargs)
#     def chat(self, messages: List[Block], tools: Optional[List[Tool]], **kwargs) -> List[Block]:
#         """Sends chat messages to the LLM with functions from the supplied tools in a side-channel.

#         Supported kwargs include:
#         - `max_tokens` (controls the size of LLM responses)
#         """
#         kwargs["max_tokens"] = 10000

#         temp_file = File.create(
#             client=self.client,
#             blocks=messages,
#             tags=[Tag(kind=TagKind.GENERATION, name=GenerationTag.PROMPT_COMPLETION)],
#         )

#         options = {}
#         if len(tools) > 0:
#             functions = []
#             for tool in tools:
#                 functions.append(tool.as_openai_function())
#             options["functions"] = functions

#         if "max_tokens" in kwargs:
#             options["max_tokens"] = kwargs["max_tokens"]

#         extra = {
#             AgentLogging.LLM_NAME: "OpenAI",
#             AgentLogging.IS_MESSAGE: True,
#             AgentLogging.MESSAGE_TYPE: AgentLogging.PROMPT,
#             AgentLogging.MESSAGE_AUTHOR: AgentLogging.LLM,
#         }

#         if logging.DEBUG >= logging.root.getEffectiveLevel():
#             extra["messages"] = json.dumps(
#                 "\n".join([f"[{msg.chat_role}] {msg.as_llm_input()}" for msg in messages])
#             )
#             extra["tools"] = ",".join([t.name for t in tools])
#         else:
#             extra["num_messages"] = len(messages)
#             extra["num_tools"] = len(tools)

#         logging.info(f"OpenAI ChatComplete ({messages[-1].as_llm_input()})", extra=extra)

#         tool_selection_task = self.generator.generate(input_file_id=temp_file.id, options=options)
#         tool_selection_task.wait()

#         return tool_selection_task.output.blocks

DEFAULT_NAME = ""
DEFAULT_BYLINE = ""
DEFAULT_IDENTITY = """"""
DEFAULT_BEHAVIOR = """"""

SYSTEM_PROMPT = """During this chat, imagine you are an advanced AI tool for improving users' resumes. You receive markdown input of a user's resume, and you respond with an improved version, based on the following tenets:
- Less is more: Use as few words as possible without losing meaning.
- Quantify and Qualify: Include numbers, stats and concrete examples that showcase accomplishments and impact.
- Readability: Use clear, concise language with punchy sentences. Reduce the word count when possible.
- Action-oriented: Use action verbs and make use of the active voice.
- Spelling and grammar: Make sure all spelling and grammar is correct.

When you respond, only respond with the improved section in markdown format. Do not write any conversational things, like "Here is an improved version of the resume"

You ONLY reply in markdown format. Make sure that formatting in the markdown reply is consistent, and remove any redundant or irregular formatting. The final document should look polished and professional without formatting mistakes.

If you receive something other than a HTML file, reply to the user that they should ensure they are uploading a .docx file.conversational things, like "Here is an improved version of the resume section"
"""



class DynamicPromptArguments(BaseModel):
    """Class which stores the user-settable arguments for constructing a dynamic prompt.

    A few notes for programmers wishing to use this example:

    - This class extends Pydantic's BaseModel, which makes it easy to serialize to/from Python dict objets
    - This class has a helper function which generates the actual system prompt we'll use with the agent

    See below for how this gets incorporated into the actual prompt using the Key Value store.
    """

    name: str = Field(default=DEFAULT_NAME, description="The name of the AI Agent")
    byline: str = Field(
        default=DEFAULT_BYLINE, description="The byline of the AI Agent"
    )
    identity: str = Field(
        default=DEFAULT_IDENTITY,
        description="The identity of the AI Agent as a bullet list",
    )
    behavior: str = Field(
        default=DEFAULT_BEHAVIOR,
        description="The behavior of the AI Agent as a bullet list",
    )

    def to_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            name=self.name,
            byline=self.byline,
            identity=self.identity,
            behavior=self.behavior,
        )


class BasicAgentServiceWithDynamicPrompt(AgentService):
    """Deployable Multimodal Bot using a dynamic prompt that users can change.

    Comes with out of the box support for:
    - Telegram
    - Slack
    - Web Embeds
    """

    USED_MIXIN_CLASSES = [SteamshipWidgetTransport, TelegramTransport, SlackTransport]
    """USED_MIXIN_CLASSES tells Steamship what additional HTTP endpoints to register on your AgentService."""

    class BasicAgentServiceWithDynamicPromptConfig(Config):
        """Pydantic definition of the user-settable Configuration of this Agent."""

        telegram_bot_token: str = Field(
            "", description="[Optional] Secret token for connecting to Telegram"
        )

    config: BasicAgentServiceWithDynamicPromptConfig
    """The configuration block that users who create an instance of this agent will provide."""

    tools: List[Tool]
    """The list of Tools that this agent is capable of using."""

    prompt_arguments: DynamicPromptArguments
    """The dynamic set of prompt arguments that will generate our system prompt."""

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class so that Steamship can auto-generate a web UI upon agent creation time."""
        return (
            BasicAgentServiceWithDynamicPrompt.BasicAgentServiceWithDynamicPromptConfig
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Tools Setup
        # -----------

        # Tools can return text, audio, video, and images. They can store & retrieve information from vector DBs, and
        # they can be stateful -- using Key-Valued storage and conversation history.
        #
        # See https://docs.steamship.com for a full list of supported Tools.
        self.tools = []

        # Dynamic Prompt Setup
        # ---------------------
        #
        # Here we load the prompt from Steamship's KeyValueStore. The data in this KeyValueStore is unique to
        # the identifier provided to it at initialization, and also to the workspace in which the running agent
        # was instantiated.
        #
        # Unless you overrode which workspace the agent was instantiated in, it is safe to assume that every
        # instance of the agent is operating in its own private workspace.
        #
        # Here is where we load the stored prompt arguments. Then see below where we set agent.PROMPT with them.

        self.kv_store = KeyValueStore(self.client, store_identifier="my-kv-store")
        self.prompt_arguments = DynamicPromptArguments.parse_obj(
            self.kv_store.get("prompt-arguments") or {}
        )

        # Agent Setup
        # ---------------------

        # This agent's planner is responsible for making decisions about what to do for a given input.
        agent = FunctionsBasedAgent(
            tools=self.tools,
            llm=ChatOpenAI(self.client, model_name="gpt-4"),
        )

        # Here is where we override the agent's prompt to set its personality. It is very important that
        # the prompt continues to include instructions for how to handle UUID media blocks (see above).
        agent.PROMPT = self.prompt_arguments.to_system_prompt()
        self.set_default_agent(agent)

        # Communication Transport Setup
        # -----------------------------

        # Support Steamship's web client
        self.add_mixin(
            SteamshipWidgetTransport(
                client=self.client,
                agent_service=self,
            )
        )

        # # Support Slack
        # self.add_mixin(
        #     SlackTransport(
        #         client=self.client,
        #         config=SlackTransportConfig(),
        #         agent_service=self,
        #     )
        # )

        # # Support Telegram
        # self.add_mixin(
        #     TelegramTransport(
        #         client=self.client,
        #         config=TelegramTransportConfig(
        #             bot_token=self.config.telegram_bot_token
        #         ),
        #         agent_service=self,
        #     )
        # )

    @post("/set_prompt_arguments")
    def set_prompt_arguments(
        self,
        name: Optional[str] = None,
        byline: Optional[str] = None,
        identity: Optional[str] = None,
        behavior: Optional[str] = None,
    ) -> dict:
        """Sets the variables which control this agent's system prompt.

        Note that we use the arguments by name here, instead of **kwargs, so that:
         1) Steamship's web UI will auto-generate UI elements for filling in the values, and
         2) API consumers who provide extra values will receive a valiation error
        """

        # # Set prompt_arguments to the new data provided by the API caller.
        # self.prompt_arguments = DynamicPromptArguments.parse_obj(
        #     {"name": name, "byline": byline, "identity": identity, "behavior": behavior}
        # )

        # # Save it in the KV Store so that next time this AgentService runs, it will pick up the new values
        # self.kv_store.set("prompt-arguments", self.prompt_arguments.dict())

        return self.prompt_arguments.dict()
    

if __name__ == "__main__":
    logging.disable(logging.ERROR)
    AgentREPL(
        BasicAgentServiceWithDynamicPrompt,
        method="prompt",
        agent_package_config={
            "botToken": "not-a-real-token-for-local-testing",
            "paymentProviderToken": "not-a-real-token-for-local-testing",
            "n_free_messages": 10,
        },
    ).run()
