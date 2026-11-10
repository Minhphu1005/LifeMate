class DummyLLM:
    def generate_calls(self, user_text, history, user_context, available_tools,
                       env_caller=None, validator=None) -> List[dict]:
        return [];

class ReadProbeLLM(DummyLLM):
    def generate_calls(self, user_text, history, user_context, available_tools,
                       env_caller=None, validator=None) -> List[dict]:
        for t in available_tools:
            if t.get("kind") == "read" and not (t.get("parameters") or {}).get("required"):
                if env_caller:
                    env_caller(t["name"], {})
                return [{"name": t["name"], "arguments": {}}]
        return [];

    @staticmethod
    def _tool_satisfy(tool):
        no_req_params = not (t.get("parameters") or {}).get("required");
        return (t.get("kind") == "read") and no_req_params;

