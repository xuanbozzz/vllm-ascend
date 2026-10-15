# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checks for the imported standalone mixed W2/W4 expert FFN.

Run with ``python csrc/mixed_w2w4_moe_ffn/tests/test_import_contract.py``.
These checks never import the operator scripts or access the original checkout.
"""

import ast
import codecs
import hashlib
import json
import re
import unittest
from pathlib import Path, PurePosixPath

IMPORT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = IMPORT_ROOT / "SOURCE_MANIFEST.json"
EXPECTED_FILE_COUNT = 39
EXPECTED_PYTHON_COUNT = 6
EXPECTED_SHELL_COUNT = 3
CORE_ENTRIES = (
    "mixed_w2w4_moe_ffn_host.cc",
    "mixed_w2w4_moe_ffn_aic.cce",
    "mixed_w2w4_moe_ffn_aiv.cce",
    "mixed_w2w4_moe_ffn_cases.ini",
    "run_mixed_w2w4_moe_ffn_esl.sh",
    "run_mixed_w2w4_moe_ffn_case.sh",
    "run_mixed_w2w4_moe_ffn_cases.sh",
    "run_mixed_w2w4_moe_ffn_cases.py",
)


class ImportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def assert_safe_relative_path(self, value):
        self.assertIsInstance(value, str)
        self.assertTrue(value, "Manifest paths must not be empty")
        self.assertNotIn("\\", value, "Use portable POSIX relative paths")
        self.assertNotIn(":", value, "Drive paths and alternate streams are unsafe")
        self.assertNotIn("\x00", value)
        self.assertFalse(PurePosixPath(value).is_absolute())
        self.assertTrue(
            all(part not in ("", ".", "..") for part in value.split("/")),
            f"Unsafe relative path: {value!r}",
        )

    def manifest_entries(self):
        self.assertIsInstance(self.manifest, dict)
        self.assertEqual(self.manifest.get("format_version"), 1)
        entries = self.manifest.get("files")
        self.assertIsInstance(entries, list)
        self.assertEqual(len(entries), EXPECTED_FILE_COUNT)
        for entry in entries:
            self.assertIsInstance(entry, dict)
            for field in ("source_path", "path"):
                self.assertIn(field, entry)
                self.assert_safe_relative_path(entry[field])
            # Reject symlink escapes as well as lexical path traversal.
            resolved = (IMPORT_ROOT / entry["path"]).resolve()
            self.assertTrue(
                resolved.is_relative_to(IMPORT_ROOT),
                f"Imported path escapes source directory: {entry['path']}",
            )
        return entries

    def test_manifest_has_unique_safe_paths_and_valid_hashes(self):
        entries = self.manifest_entries()
        for field in ("source_path", "path"):
            paths = [entry[field] for entry in entries]
            self.assertEqual(len(set(paths)), EXPECTED_FILE_COUNT)
            self.assertEqual(len({path.casefold() for path in paths}), EXPECTED_FILE_COUNT)
        for entry in entries:
            for field in ("source_sha256", "sha256"):
                with self.subTest(path=entry["path"], field=field):
                    self.assertIn(field, entry)
                    self.assertIsInstance(entry[field], str)
                    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{64}", entry[field]))

    def test_imported_files_are_nonempty_and_match_manifest(self):
        for entry in self.manifest_entries():
            with self.subTest(path=entry["path"]):
                path = IMPORT_ROOT / entry["path"]
                self.assertTrue(path.is_file())
                data = path.read_bytes()
                self.assertTrue(data, "Imported files must not be empty")
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def test_all_six_python_sources_parse_without_importing(self):
        entries = [entry for entry in self.manifest_entries() if PurePosixPath(entry["path"]).suffix == ".py"]
        self.assertEqual(len(entries), EXPECTED_PYTHON_COUNT)
        for entry in entries:
            with self.subTest(path=entry["path"]):
                path = IMPORT_ROOT / entry["path"]
                ast.parse(path.read_bytes(), filename=entry["path"])

    def test_all_three_shell_scripts_use_lf_without_bom(self):
        entries = [entry for entry in self.manifest_entries() if PurePosixPath(entry["path"]).suffix == ".sh"]
        self.assertEqual(len(entries), EXPECTED_SHELL_COUNT)
        for entry in entries:
            with self.subTest(path=entry["path"]):
                data = (IMPORT_ROOT / entry["path"]).read_bytes()
                self.assertNotIn(b"\r", data)
                for bom in (
                    codecs.BOM_UTF8,
                    codecs.BOM_UTF16_LE,
                    codecs.BOM_UTF16_BE,
                    codecs.BOM_UTF32_LE,
                    codecs.BOM_UTF32_BE,
                ):
                    self.assertFalse(data.startswith(bom), "Unexpected BOM")
                self.assertTrue(data.startswith(b"#!/usr/bin/env bash\n"))
                self.assertTrue(data.endswith(b"\n"))

    def test_core_entry_points_and_uniform_operator_name(self):
        entries = self.manifest_entries()
        imported_paths = {entry["path"] for entry in entries}
        self.assertEqual(IMPORT_ROOT.name, "mixed_w2w4_moe_ffn")
        for entry_point in CORE_ENTRIES:
            with self.subTest(entry_point=entry_point):
                self.assertIn(entry_point, imported_paths)
                self.assertTrue((IMPORT_ROOT / entry_point).is_file())
        for entry in entries:
            with self.subTest(path=entry["path"]):
                self.assertNotIn("mat_fp4", entry["path"])
                self.assertNotIn(b"mat_fp4", (IMPORT_ROOT / entry["path"]).read_bytes())
                self.assertNotIn("matmul_W2A8", entry["path"])
                self.assertNotIn(b"matmul_W2A8", (IMPORT_ROOT / entry["path"]).read_bytes())

    def test_runtime_and_build_names_match_device_entry_points(self):
        operator_name = IMPORT_ROOT.name
        host = (IMPORT_ROOT / f"{operator_name}_host.cc").read_text(encoding="utf-8")
        build = (IMPORT_ROOT / f"run_{operator_name}_esl.sh").read_text(encoding="utf-8")
        runner = (IMPORT_ROOT / f"run_{operator_name}_case.sh").read_text(encoding="utf-8")
        self.assertIn(f'kKernelName = "{operator_name}";', host)
        self.assertIn(f'kKernelBinary = "{operator_name}.o";', host)
        self.assertIn(f"-o {operator_name}.o", build)
        self.assertIn(f"{operator_name}_host.cc", build)
        self.assertIn(f"-o {operator_name}_host", build)
        self.assertIn(f'--application="./{operator_name}_host $*"', runner)
        for core in ("aic", "aiv"):
            with self.subTest(core=core):
                entry_point = f"{operator_name}_mix_{core}"
                source = IMPORT_ROOT / f"{operator_name}_{core}.cce"
                self.assertIn(f"void {entry_point}(", source.read_text(encoding="utf-8"))
                self.assertIn(f"{source.name} {entry_point}.o", build)
                for component in ("gate_w2", "gate_w4", "down_w2", "down_w4"):
                    header = IMPORT_ROOT / "components" / component / f"{component}_{core}_impl.h"
                    self.assertIn(f"void {entry_point}(", header.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
