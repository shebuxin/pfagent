import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.andes_manual import BASE_ANDES_MANUAL_POLICY, RAG_ANDES_MANUAL_POLICY
from src.prompt_builder import AndesPromptBuilderConfig, AndesSystemPromptBuilder
from src.chatbots.openai.simple_chatbot import SimpleChatbot, SimpleChatConfig
from src.chatbots.openai.rag_chatbot import RAGChatbot, RAGConfig
from src.chatbots.openai.graphrag_chatbot import GraphRAGChatbot, GraphRAGConfig


class PromptBuilderTests(unittest.TestCase):
    def test_simple_prompt_omits_context_and_tools_sections(self):
        builder = AndesSystemPromptBuilder(
            AndesPromptBuilderConfig(
                include_context_placeholder=False,
                include_tools_info=False,
                enforce_code_only_fence_rule=False,
                include_andes_guardrails=False,
                ban_test_wrappers=False,
            )
        )

        prompt = builder.build_prompt(
            andes_manual_policy=BASE_ANDES_MANUAL_POLICY,
            tools_info="TOOL_INFO_SHOULD_NOT_APPEAR",
            custom_instructions="Be concise.",
            few_shot_section="FEW_SHOT_SECTION",
        )

        self.assertNotIn("<context>", prompt)
        self.assertNotIn("TOOL_INFO_SHOULD_NOT_APPEAR", prompt)
        self.assertNotIn("ANDES 2.0 Guardrails", prompt)
        self.assertIn("FEW_SHOT_SECTION", prompt)
        self.assertIn("Additional Instructions:\nBe concise.", prompt)

    def test_rag_prompt_includes_context_tools_and_guardrails(self):
        builder = AndesSystemPromptBuilder(
            AndesPromptBuilderConfig(
                include_context_placeholder=True,
                include_tools_info=True,
                enforce_code_only_fence_rule=True,
                include_andes_guardrails=True,
                ban_test_wrappers=True,
            )
        )

        prompt = builder.build_prompt(
            andes_manual_policy=RAG_ANDES_MANUAL_POLICY,
            tools_info="TOOLS_SECTION",
            custom_instructions="Use the uploaded case preview.",
            few_shot_section="FEW_SHOT_SECTION",
        )

        self.assertIn("<context>", prompt)
        self.assertIn("TOOLS_SECTION", prompt)
        self.assertIn("5a. If the user asks for runnable Python code only", prompt)
        self.assertIn("answer in plain prose and do not emit Python", prompt)
        self.assertIn("ANDES 2.0 Guardrails", prompt)
        self.assertIn("`ssa.Line.idx.v` contains string device IDs", prompt)
        self.assertIn("`ssa.Line.a1.e` / `ssa.Line.a2.e`", prompt)
        self.assertIn(
            '`ssa.Line.set(src="u", idx=line_id, attr="v", value=0)`',
            prompt,
        )
        self.assertIn("backward-compatible across ANDES 1.x and 2.0", prompt)
        self.assertIn("Pass `idx` and `value` as scalars, NOT as lists", prompt)
        self.assertIn("`ssa.PFlow.converged`", prompt)
        self.assertIn("`ssa.Bus.nosw_island`", prompt)
        self.assertIn("Do not wrap the answer in unittest, pytest", prompt)
        self.assertIn("FEW_SHOT_SECTION", prompt)
        self.assertIn("Additional Instructions:\nUse the uploaded case preview.", prompt)

    def test_chatbots_load_prompts_from_shared_builder(self):
        simple = SimpleChatbot(SimpleChatConfig(openai_api_key="test"))
        simple.load_system_prompt(custom_instructions="Simple prompt custom line.")

        rag = RAGChatbot(RAGConfig(openai_api_key="test"))
        rag.load_system_prompt(custom_instructions="RAG prompt custom line.")

        with patch("src.chatbots.openai.graphrag_chatbot.Neo4jGraphStore"):
            graph = GraphRAGChatbot(
                GraphRAGConfig(
                    openai_api_key="test",
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="password",
                )
            )
        graph.load_system_prompt(custom_instructions="Graph prompt custom line.")

        shared_phrase = "ANDES Case Modification Rules:"
        self.assertIn(shared_phrase, simple.system_message)
        self.assertIn(shared_phrase, rag.system_message)
        self.assertIn(shared_phrase, graph.system_message)

        self.assertNotIn("<context>", simple.system_message)
        self.assertIn("<context>", rag.system_message)
        self.assertIn("<context>", graph.system_message)

        self.assertIn("Simple prompt custom line.", simple.system_message)
        self.assertIn("RAG prompt custom line.", rag.system_message)
        self.assertIn("Graph prompt custom line.", graph.system_message)

        self.assertIn("ANDES 2.0 Guardrails", rag.system_message)
        self.assertNotIn("ANDES 2.0 Guardrails", simple.system_message)
        self.assertNotIn("ANDES 2.0 Guardrails", graph.system_message)


if __name__ == "__main__":
    unittest.main()
