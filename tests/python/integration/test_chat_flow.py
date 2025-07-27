


def test_chat_flow(client, chat_context_manager):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert "response" in response.json()