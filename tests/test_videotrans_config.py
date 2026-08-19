import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from videotrans import config


class DotEnvTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        config._dotenv_loaded_paths.clear()
        # Strip every key a .env file may set so tests are hermetic.
        env = {k: v for k, v in os.environ.items()
               if k != "DASHSCOPE_API_KEY" and not k.startswith("OIL_SUBTITLE_")}
        self._env = mock.patch.dict(os.environ, env, clear=True)
        self._env.start()
        self.addCleanup(config._dotenv_loaded_paths.clear)

    def _chdir(self, path: Path):
        previous = Path.cwd()
        os.chdir(path)
        self.addCleanup(lambda: os.chdir(previous))

    def test_api_key_read_from_local_env_file(self):
        (self.tmp / ".env").write_text(
            "DASHSCOPE_API_KEY=sk-from-dotenv-123\n", encoding="utf-8"
        )
        self._chdir(self.tmp)
        self.assertEqual(
            config.load_dashscope_api_key(required=False), "sk-from-dotenv-123"
        )

    def test_real_environment_beats_env_file(self):
        (self.tmp / ".env").write_text(
            "DASHSCOPE_API_KEY=sk-from-dotenv\n", encoding="utf-8"
        )
        self._chdir(self.tmp)
        os.environ["DASHSCOPE_API_KEY"] = "sk-from-real-env"
        self.assertEqual(
            config.load_dashscope_api_key(required=False), "sk-from-real-env"
        )

    def test_parser_handles_quotes_comments_and_junk(self):
        env_file = self.tmp / ".env"
        env_file.write_text(
            "# comment line\n"
            "\n"
            "DASHSCOPE_API_KEY=\"sk-quoted\"\n"
            "OIL_SUBTITLE_PROGRESS_ENABLED=on\n"
            "not-a-pair\n"
            "EMPTY=\n",
            encoding="utf-8",
        )
        loaded = config.load_env_file(env_file)
        self.assertEqual(loaded["DASHSCOPE_API_KEY"], "sk-quoted")
        self.assertEqual(loaded["OIL_SUBTITLE_PROGRESS_ENABLED"], "on")
        self.assertNotIn("not-a-pair", loaded)
        self.assertEqual(os.environ.get("EMPTY"), "")

    def test_missing_env_file_is_noop(self):
        self._chdir(self.tmp)
        self.assertEqual(config.load_env_file(), {})
        # No .env, no env var, no key-file fallback anymore: empty result.
        self.assertEqual(config.load_dashscope_api_key(required=False), "")

    def test_missing_key_raises_with_hint_when_required(self):
        self._chdir(self.tmp)
        with self.assertRaises(RuntimeError) as ctx:
            config.load_dashscope_api_key(required=True)
        self.assertIn(".env", str(ctx.exception))

    def test_env_file_values_reach_other_resolvers(self):
        (self.tmp / ".env").write_text(
            "OIL_SUBTITLE_PROGRESS_ENABLED=off\n", encoding="utf-8"
        )
        self._chdir(self.tmp)
        # Trigger the lazy loader, then resolve as the pipeline would.
        config.env_value("OIL_SUBTITLE_PROGRESS_ENABLED")
        with mock.patch.object(config, "load_user_config", return_value={}):
            self.assertFalse(config.resolve_progress_enabled())
        with mock.patch.object(config, "load_user_config", return_value={}):
            self.assertTrue(config.resolve_progress_enabled(True))


class WordlistResolutionTests(unittest.TestCase):
    """Hotwords/glossary resolve from .env entries or CLI flags only."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        previous = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(lambda: os.chdir(previous))
        config._dotenv_loaded_paths.clear()
        self.addCleanup(config._dotenv_loaded_paths.clear)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("OIL_SUBTITLE_")}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_glossary_defaults_to_project_local_file(self):
        self.assertEqual(config.resolve_glossary_path(), Path("glossary.json"))

    def test_glossary_env_override(self):
        os.environ["OIL_SUBTITLE_GLOSSARY"] = str(self.tmp / "my-glossary.json")
        self.assertEqual(
            config.resolve_glossary_path(), self.tmp / "my-glossary.json"
        )

    def test_glossary_cli_override_beats_env(self):
        os.environ["OIL_SUBTITLE_GLOSSARY"] = str(self.tmp / "from-env.json")
        self.assertEqual(
            config.resolve_glossary_path("from-cli.json"), Path("from-cli.json")
        )

    def test_hotwords_disabled_when_unconfigured(self):
        self.assertIsNone(config.resolve_hotwords_path())

    def test_hotwords_env_override(self):
        os.environ["OIL_SUBTITLE_HOTWORDS"] = str(self.tmp / "hotwords.json")
        self.assertEqual(
            config.resolve_hotwords_path(), self.tmp / "hotwords.json"
        )


class SaveApiKeyTests(unittest.TestCase):
    """--save-api-key upserts DASHSCOPE_API_KEY in the local .env."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        previous = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(lambda: os.chdir(previous))
        env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_creates_env_file_when_absent(self):
        target = config.save_dashscope_api_key("sk-new")
        self.assertEqual(target, self.tmp / ".env")
        content = (self.tmp / ".env").read_text(encoding="utf-8")
        self.assertIn("DASHSCOPE_API_KEY=sk-new", content)

    def test_replaces_existing_key_and_keeps_other_lines(self):
        (self.tmp / ".env").write_text(
            "# comment\nDASHSCOPE_API_KEY=sk-old\nOIL_SUBTITLE_GLOSSARY=./g.json\n",
            encoding="utf-8",
        )
        config.save_dashscope_api_key("sk-rotated")
        content = (self.tmp / ".env").read_text(encoding="utf-8")
        self.assertIn("DASHSCOPE_API_KEY=sk-rotated", content)
        self.assertNotIn("sk-old", content)
        self.assertIn("# comment", content)
        self.assertIn("OIL_SUBTITLE_GLOSSARY=./g.json", content)
        # And the rotated key is what the loader now returns.
        config._dotenv_loaded_paths.clear()
        self.assertEqual(config.load_dashscope_api_key(required=False), "sk-rotated")

    def test_leaves_commented_example_untouched_and_appends(self):
        (self.tmp / ".env").write_text(
            "# DASHSCOPE_API_KEY=sk-your-key-here\n", encoding="utf-8"
        )
        config.save_dashscope_api_key("sk-real")
        content = (self.tmp / ".env").read_text(encoding="utf-8")
        self.assertIn("# DASHSCOPE_API_KEY=sk-your-key-here", content)
        self.assertIn("DASHSCOPE_API_KEY=sk-real", content)

    def test_empty_key_is_rejected(self):
        with self.assertRaises(ValueError):
            config.save_dashscope_api_key("  ")


if __name__ == "__main__":
    unittest.main()
