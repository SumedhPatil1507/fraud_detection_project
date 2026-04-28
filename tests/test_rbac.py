import pytest
from src.rbac import Role, has_permission, PERMISSIONS


def test_admin_has_all_permissions():
    assert has_permission(Role.ADMIN, "predict")
    assert has_permission(Role.ADMIN, "train")
    assert has_permission(Role.ADMIN, "manage_users")


def test_analyst_cannot_train():
    assert has_permission(Role.ANALYST, "predict")
    assert not has_permission(Role.ANALYST, "train")
    assert not has_permission(Role.ANALYST, "manage_users")


def test_viewer_read_only():
    assert has_permission(Role.VIEWER, "model_info")
    assert not has_permission(Role.VIEWER, "predict")
    assert not has_permission(Role.VIEWER, "train")


def test_all_roles_defined():
    for role in Role:
        assert role in PERMISSIONS
