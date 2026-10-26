"""Orchestrate a single isolated conversation without passing history as facts."""

class ChatService:
    def __init__(self, session_store, pipeline, memory_messages=8):
        self.sessions, self.pipeline, self.memory_messages = session_store, pipeline, memory_messages

    def ask(self, session_id: str, question: str):
        history = self.sessions.get_messages(session_id, limit=self.memory_messages)
        answer = self.pipeline.answer(question, history)
        self.sessions.save_turn(session_id, question, answer.text, answer.citations)
        return answer
