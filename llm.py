from openai import OpenAI

client = OpenAI()          # reads OPENAI_API_KEY from the environment
MODEL = "gpt-4o-mini"      # change if your account uses a different model


def chat(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return resp.choices[0].message.content