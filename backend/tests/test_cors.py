def test_preflight_is_cached(sqlite_client):
    response = sqlite_client.options(
        "/api/conversations",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-max-age"] == "86400"
