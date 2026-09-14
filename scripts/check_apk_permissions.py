"""Assert the built APK grants only the permissions we intend.

    python scripts/check_apk_permissions.py --apk <path>
    python scripts/check_apk_permissions.py            # finds the debug APKs

The source manifest is not the answer to "what does this app ask for". Android's
manifest merger unions in every dependency's manifest, so a library can add a
permission the app never requested and the source file will not show it.

That is not hypothetical here. `onnxruntime-android` declares INTERNET and
ACCESS_NETWORK_STATE, so before an explicit `tools:node="remove"` the shipped
APK carried both -- in an app whose entire claim is that your photograph cannot
leave the phone. The claim was false in the artefact while true in the source.

So this checks the artefact. `PRIVACY.md` tells users they can verify the
permission set themselves; this is the same check, run in CI.

Exit code 1 on any unexpected permission, so it can gate a release.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Everything the app is allowed to ask for, with why. Adding to this list
#: should be a conscious, reviewed act -- that is the point of the list.
ALLOWED = {
    "android.permission.CAMERA": "photographing a fish",
    "android.permission.READ_MEDIA_IMAGES": "importing a photo you picked (API 33+)",
    "android.permission.READ_EXTERNAL_STORAGE": "the same, on API <= 32",
    "android.permission.ACCESS_COARSE_LOCATION": "optional on-device geographic ranking",
    "android.permission.ACCESS_FINE_LOCATION": "the same, if the user grants precise",
}

#: Permissions that must never appear, with the reason they are called out
#: rather than merely absent from ALLOWED.
FORBIDDEN = {
    "android.permission.INTERNET":
        "the app has no network code; onnxruntime-android injects this",
    "android.permission.ACCESS_NETWORK_STATE":
        "same source, same reason",
    "android.permission.ACCESS_BACKGROUND_LOCATION":
        "location is read once, on demand, never in the background",
    "android.permission.READ_CONTACTS": "never",
    "android.permission.GET_ACCOUNTS": "never; there are no accounts",
    "android.permission.READ_PHONE_STATE": "never",
    "android.permission.RECORD_AUDIO": "never",
}

# --- minimal binary-XML reader -------------------------------------------
# Enough of the AXML format to read <uses-permission android:name="...">.
# aapt2 would do this, but it is not always on PATH and this keeps the check
# runnable from a bare checkout.

_STRING_POOL = 0x0001
_START_ELEMENT = 0x0102


def _parse_string_pool(data: bytes, off: int) -> list[str]:
    string_count, _style_count, flags, strings_start, _ = struct.unpack_from(
        "<IIIII", data, off + 8
    )
    is_utf8 = bool(flags & (1 << 8))
    offsets = struct.unpack_from(f"<{string_count}I", data, off + 28)
    base = off + strings_start
    out = []
    for o in offsets:
        p = base + o
        if is_utf8:
            # two varint lengths (chars, bytes); high bit marks a two-byte form
            n = data[p]
            p += 2 if n & 0x80 else 1
            n = data[p]
            if n & 0x80:
                n = ((n & 0x7F) << 8) | data[p + 1]
                p += 2
            else:
                p += 1
            out.append(data[p:p + n].decode("utf-8", "replace"))
        else:
            n = struct.unpack_from("<H", data, p)[0]
            if n & 0x8000:
                n = ((n & 0x7FFF) << 16) | struct.unpack_from("<H", data, p + 2)[0]
                p += 4
            else:
                p += 2
            out.append(data[p:p + n * 2].decode("utf-16-le", "replace"))
    return out


def uses_permissions(axml: bytes) -> list[str]:
    """Names declared by <uses-permission> elements, in document order."""
    strings: list[str] = []
    off = 8  # skip the file header
    found: list[str] = []
    while off + 8 <= len(axml):
        chunk_type, header_size, chunk_size = struct.unpack_from("<HHI", axml, off)
        if chunk_size <= 0 or off + chunk_size > len(axml):
            break
        if chunk_type == _STRING_POOL:
            strings = _parse_string_pool(axml, off)
        elif chunk_type == _START_ELEMENT and strings:
            ns_i, name_i = struct.unpack_from("<iI", axml, off + header_size)
            del ns_i
            if name_i < len(strings) and strings[name_i] == "uses-permission":
                attr_start, attr_size, attr_count = struct.unpack_from(
                    "<HHH", axml, off + header_size + 8
                )
                for a in range(attr_count):
                    ao = off + header_size + attr_start + a * attr_size
                    _ns, an, raw = struct.unpack_from("<iiI", axml, ao)
                    if an < len(strings) and strings[an] == "name":
                        # attribute value: prefer the raw string, fall back to
                        # the typed data slot for the same index
                        if raw != 0xFFFFFFFF and raw < len(strings):
                            found.append(strings[raw])
                        else:
                            data_i = struct.unpack_from("<I", axml, ao + 16)[0]
                            if data_i < len(strings):
                                found.append(strings[data_i])
        off += chunk_size
    return found


#: Permissions an app defines for itself, namespaced under its own package.
#: androidx.core generates one of these for RECEIVER_NOT_EXPORTED compatibility.
#: They are signature-level and scoped to this app, so they grant no access to
#: anything outside it -- but they are matched by prefix rather than waved
#: through, so a dependency cannot smuggle one in under a different package.
SELF_DEFINED_PREFIX = "com.fisherwiki.app"


def classify(perm: str) -> tuple[bool, str]:
    """(is_acceptable, explanation)."""
    if perm in FORBIDDEN:
        return False, FORBIDDEN[perm]
    if perm in ALLOWED:
        return True, ALLOWED[perm]
    if perm.startswith(SELF_DEFINED_PREFIX + "."):
        return True, "self-defined, signature-level, scoped to this app"
    return False, ("not in the allow-list; if it is intended, add it to "
                   "ALLOWED with a reason")


def check(apk: Path) -> int:
    with zipfile.ZipFile(apk) as z:
        axml = z.read("AndroidManifest.xml")
    perms = sorted(set(uses_permissions(axml)))
    if not perms:
        print(f"  FAIL  parsed no permissions from {apk.name}; the check itself "
              f"is broken and must not be read as a pass")
        return 1

    print(f"{apk.name}  ({apk.stat().st_size / 1e6:.1f} MB)")
    problems = []
    for p in perms:
        acceptable, why = classify(p)
        print(f"  {'ok  ' if acceptable else 'FAIL'}  {p}  ({why})")
        if not acceptable:
            problems.append(p)
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apk", action="append", default=[])
    args = ap.parse_args(argv)

    apks = [Path(a) for a in args.apk]
    if not apks:
        apks = sorted((REPO / "app" / "android" / "build" / "outputs" / "apk")
                      .rglob("*.apk"))
    if not apks:
        print("no APK found; run ./gradlew :android:assembleDebug first")
        return 1

    rc = 0
    for apk in apks:
        rc |= check(apk)
        print()
    print("permission set verified" if rc == 0 else "PERMISSION CHECK FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
