import unittest

from src.mm_toolbox.websocket import VerifyWsPayload


class TestVerifyWsPayload(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey1": True, "subsubkey2": None},
            },
        }
        self.verifier_strict = VerifyWsPayload(self.sample, strict=True)
        self.verifier_non_strict = VerifyWsPayload(self.sample, strict=False)

    def test_exact_match_non_strict(self):
        payload = {
            "key1": "value1",
            "key2": "123",  # Allowed in non-strict mode
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": "456.78",  # Allowed in non-strict mode
                "subkey3": {
                    "subsubkey1": "True",  # Allowed in non-strict mode
                    "subsubkey2": "None",  # Allowed in non-strict mode
                },
            },
        }
        self.assertTrue(self.verifier_non_strict.verify(payload))

    def test_exact_match_strict(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey1": True, "subsubkey2": None},
            },
        }
        self.assertTrue(self.verifier_strict.verify(payload))

    def test_type_mismatch_strict(self):
        payload = {
            "key1": "value1",
            "key2": "123",  # Should fail in strict mode
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey1": True, "subsubkey2": None},
            },
        }
        self.assertFalse(self.verifier_strict.verify(payload))

    def test_missing_key(self):
        payload = {
            "key1": "value1",
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey1": True, "subsubkey2": None},
            },
        }
        self.assertFalse(self.verifier_strict.verify(payload))
        self.assertFalse(self.verifier_non_strict.verify(payload))

    def test_extra_key(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey1": True, "subsubkey2": None},
            },
            "extra_key": "extra_value",
        }
        self.assertTrue(self.verifier_strict.verify(payload))  # Extra keys ignored
        self.assertTrue(self.verifier_non_strict.verify(payload))  # Extra keys ignored

    def test_nested_type_mismatch_strict(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {
                    "subsubkey1": "True",  # Should fail in strict mode
                    "subsubkey2": None,
                },
            },
        }
        self.assertFalse(self.verifier_strict.verify(payload))

    def test_deeply_nested_missing_key(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456.78,
                "subkey3": {"subsubkey2": None},  # "subsubkey1" is missing
            },
        }
        self.assertFalse(self.verifier_strict.verify(payload))
        self.assertFalse(self.verifier_non_strict.verify(payload))

    def test_empty_payload(self):
        payload = {}
        self.assertFalse(self.verifier_strict.verify(payload))
        self.assertFalse(self.verifier_non_strict.verify(payload))

    def test_empty_sample(self):
        empty_sample = {}
        verifier_strict_empty = VerifyWsPayload(empty_sample, strict=True)
        verifier_non_strict_empty = VerifyWsPayload(empty_sample, strict=False)

        self.assertTrue(verifier_strict_empty.verify({}))  # Both should pass
        self.assertTrue(verifier_non_strict_empty.verify({}))

    def test_payload_with_empty_dict(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {"subkey1": "subvalue1", "subkey2": 456.78, "subkey3": {}},
        }
        self.assertFalse(self.verifier_strict.verify(payload))
        self.assertFalse(self.verifier_non_strict.verify(payload))

    def test_payload_with_mixed_types(self):
        payload = {
            "key1": "value1",
            "key2": 123,
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": 456,
                "subkey3": {"subsubkey1": False, "subsubkey2": "None"},
            },
        }
        self.assertFalse(self.verifier_strict.verify(payload))  # Strict mode fails
        self.assertTrue(
            self.verifier_non_strict.verify(payload)
        )  # Non-strict mode passes


if __name__ == "__main__":
    unittest.main()
