def test_admin_models_requires_login(client):
    """Unauthenticated requests to admin endpoints should be redirected/denied."""
    resp = client.get("/admin/models")
    assert resp.status_code in (302, 401, 403)


def test_admin_export_requires_login(client):
    resp = client.get("/admin/models/export/pdf")
    assert resp.status_code in (302, 401, 403)
