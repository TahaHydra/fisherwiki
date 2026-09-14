"""The permission check must be able to fail, and must agree with the merger.

A privacy claim backed by a check that always passes is worse than no check:
it converts an unverified assertion into an apparently verified one. These
tests exist to keep `scripts/check_apk_permissions.py` capable of catching the
thing it was written for -- `onnxruntime-android` silently adding INTERNET to
an app that has no network code.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from check_apk_permissions import (  # noqa: E402
    ALLOWED,
    FORBIDDEN,
    classify,
    uses_permissions,
)

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
APK_DIR = REPO / "app" / "android" / "build" / "outputs" / "apk"
MERGED = (REPO / "app" / "android" / "build" / "intermediates" / "merged_manifest"
          / "debug" / "processDebugMainManifest" / "AndroidManifest.xml")


# --------------------------------------------------------------- classify

def test_internet_is_rejected():
    ok, why = classify("android.permission.INTERNET")
    assert not ok
    assert "onnxruntime" in why


def test_network_state_is_rejected():
    assert not classify("android.permission.ACCESS_NETWORK_STATE")[0]


def test_background_location_is_rejected():
    assert not classify("android.permission.ACCESS_BACKGROUND_LOCATION")[0]


def test_an_unknown_permission_is_rejected_rather_than_ignored():
    # The failure mode to avoid is a new dependency adding something nobody
    # listed, and the check shrugging because it is not on the deny-list.
    ok, why = classify("android.permission.SEND_SMS")
    assert not ok
    assert "allow-list" in why


def test_camera_is_allowed():
    ok, why = classify("android.permission.CAMERA")
    assert ok and "fish" in why


def test_a_self_defined_permission_is_allowed():
    ok, why = classify("com.fisherwiki.app.debug.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION")
    assert ok and "signature-level" in why


def test_another_package_cannot_pose_as_self_defined():
    # Prefix matching must not let com.evil.fisherwiki.app.* through, nor a
    # package that merely starts with the same characters.
    assert not classify("com.evil.com.fisherwiki.app.THING")[0]
    assert not classify("com.fisherwiki.appx.THING")[0]


def test_allow_and_forbid_lists_do_not_overlap():
    assert not (set(ALLOWED) & set(FORBIDDEN))


def test_every_entry_carries_a_reason():
    for table in (ALLOWED, FORBIDDEN):
        for name, reason in table.items():
            assert reason.strip(), f"{name} has no stated reason"


# ------------------------------------------------------------ APK parsing

@pytest.mark.skipif(not MERGED.exists(),
                    reason="no merged manifest; run ./gradlew :android:assembleDebug")
def test_the_axml_parser_agrees_with_the_merged_manifest():
    """The binary parser is hand-rolled, so check it against Gradle's own XML.

    If these disagree the check is reading something other than what ships,
    which would make a pass meaningless.
    """
    apks = sorted(APK_DIR.rglob("*debug.apk"))
    if not apks:
        pytest.skip("no APK built")

    expected = sorted({e.get(ANDROID_NS + "name")
                       for e in ET.parse(MERGED).getroot().findall("uses-permission")})
    with zipfile.ZipFile(apks[0]) as z:
        actual = sorted(set(uses_permissions(z.read("AndroidManifest.xml"))))
    assert actual == expected


@pytest.mark.skipif(not APK_DIR.exists(),
                    reason="no APK; run ./gradlew :android:assembleDebug")
def test_no_built_apk_grants_network_access():
    """The claim in PRIVACY.md, asserted against the artefact users install."""
    apks = sorted(APK_DIR.rglob("*.apk"))
    if not apks:
        pytest.skip("no APK built")
    for apk in apks:
        with zipfile.ZipFile(apk) as z:
            perms = set(uses_permissions(z.read("AndroidManifest.xml")))
        assert perms, f"{apk.name}: parsed no permissions at all"
        assert "android.permission.INTERNET" not in perms, apk.name
        assert "android.permission.ACCESS_NETWORK_STATE" not in perms, apk.name
