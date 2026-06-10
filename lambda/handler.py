"""
Lambda handler (placeholder)

hardcoded text to validate API Gateway - Lambda connection, before integration with Bedrock

the message (event, context - dict con statusCode/headers/body) is the payload waited. DON't MAKE CHANGES HERE
"""
import json


def lambda_handler(event, context):
    # event["body"] = string (or None if body doesn't exists).
    body = json.loads(event.get("body") or "{}")
    question = body.get("question", "(no question provided)")

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({
            "answer": (
                f"Phase 1 placeholder. Received question: {question!r}. "
                "Bedrock wiring lands in Phase 1.2."
            ),
            "phase": "1",
        }),
    }
