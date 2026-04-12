import sys
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.conversation_compaction import (
    ConversationCompactionConfig,
    build_compacted_message_window,
    build_readable_conversation_summary,
)
from src.chatbots.openai.rag_chatbot import RAGChatbot, RAGConfig


class ConversationCompactionTests(unittest.TestCase):
    def test_compaction_inserts_summary_and_keeps_recent_messages(self):
        history = [
            HumanMessage(content="Use my uploaded case uploaded_ieee39.xlsx and return runnable Python code only."),
            AIMessage(content="```python\nimport andes\nssa = andes.load('uploaded_ieee39.xlsx', setup=True)\n```"),
            HumanMessage(content="Modify the PV at bus 35. Resolve the real idx from the case and do not guess idx."),
            AIMessage(content="```python\npv_idx = ssa.PV.idx.v[0]\nssa.PV.set(src='v0', attr='v', idx=pv_idx, value=1.03)\n```"),
            HumanMessage(content="Print RESULT_JSON and keep the exact uploaded filename."),
            AIMessage(content="```python\nimport json\nRESULT_JSON = {'ok': True}\nprint('RESULT_JSON=' + json.dumps(RESULT_JSON))\n```"),
        ]
        config = ConversationCompactionConfig(
            trigger_message_count=4,
            keep_recent_messages=1,
            max_summary_chars=2200,
        )

        messages, summary = build_compacted_message_window(
            system_message="You are PFAGENT.",
            conversation_history=history,
            config=config,
        )

        self.assertEqual(3, len(messages))
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], SystemMessage)
        self.assertEqual(history[-1:], messages[-1:])
        self.assertIn("uploaded_ieee39.xlsx", summary)
        self.assertIn("Return one runnable Python script only.", summary)
        self.assertIn("Preserve RESULT_JSON output requirements when requested.", summary)
        self.assertIn("Resolve real device idx values from the case before calling .set(...).", summary)

    def test_below_threshold_keeps_full_history_without_summary(self):
        history = [
            HumanMessage(content="Use ieee14/ieee14_full.xlsx."),
            AIMessage(content="I can help with that."),
            HumanMessage(content="Report the top-3 voltages."),
        ]
        config = ConversationCompactionConfig(
            trigger_message_count=5,
            keep_recent_messages=2,
        )

        messages, summary = build_compacted_message_window(
            system_message="You are PFAGENT.",
            conversation_history=history,
            config=config,
        )

        self.assertEqual("", summary)
        self.assertEqual(4, len(messages))
        self.assertEqual(history, messages[1:])

    def test_readable_summary_surfaces_compacted_and_recent_state(self):
        history = [
            HumanMessage(content="Use built-in case ieee14/ieee14_full.xlsx."),
            AIMessage(content="I will load the built-in case."),
            HumanMessage(content="Then add one PQ load before setup."),
            AIMessage(content="I will add the device before setup."),
            HumanMessage(content="Finally plot the voltage profile."),
            AIMessage(content="I will save the plot."),
        ]
        config = ConversationCompactionConfig(
            trigger_message_count=4,
            keep_recent_messages=2,
        )

        summary = build_readable_conversation_summary(history, config)
        self.assertIn("Compacted earlier session", summary)
        self.assertIn("ieee14/ieee14_full.xlsx", summary)
        self.assertIn("Recent live messages", summary)
        self.assertIn("plot the voltage profile", summary)

    def test_rag_chatbot_builds_compacted_window_for_long_sessions(self):
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key="test",
                chat_model="gpt-4.1-mini",
                conversation_compaction_trigger_messages=4,
                conversation_compaction_keep_recent_messages=2,
                conversation_compaction_max_summary_chars=2200,
            )
        )
        chatbot.conversation_history = [
            HumanMessage(content="Use uploaded_ieee39.xlsx."),
            AIMessage(content="I will use the uploaded case."),
            HumanMessage(content="Modify the PV at bus 35 by resolving idx from the case."),
            AIMessage(content="I will resolve idx before calling .set(...)."),
            HumanMessage(content="Return runnable Python code only and print RESULT_JSON."),
        ]

        messages = chatbot._build_model_messages("System prompt")

        self.assertEqual(4, len(messages))
        self.assertTrue(chatbot.last_compaction_summary)
        self.assertIn("uploaded_ieee39.xlsx", chatbot.last_compaction_summary)
        self.assertEqual("Return runnable Python code only and print RESULT_JSON.", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
