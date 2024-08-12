from typing import Dict, Any, Union


class VerifyWsPayload:
    """
    A class to verify if a given payload matches the structure of a sample.

    Parameters
    ----------
    sample : Dict
        The sample dictionary to use as a reference for structure and keys.

    strict : bool, optional
        If True, enforce strict type checking on the values of the keys (default is False).
    """
    def __init__(self, sample: Dict, strict: bool = False) -> None:
        self.sample = sample
        self.strict = strict
        self.type_map = self._create_type_map_(sample)
    
    def verify(self, payload: Dict) -> bool:
        """
        Check if the payload matches the structure of the sample.

        Parameters
        ----------
        payload : Dict
            The payload dictionary to verify.

        Returns
        -------
        bool
            True if the payload matches the sample structure, False otherwise.
        """
        return self._check_structure_(self.type_map, payload)
    
    def _create_type_map_(self, sample: Dict) -> Dict:
        """
        Creates a type map for the sample dictionary.

        Parameters
        ----------
        sample : Dict
            The sample dictionary to use as a reference for structure and keys.

        Returns
        -------
        Dict
            A dictionary with the same structure as the sample but with types as values.
        """
        type_map = {}

        for key, value in sample.items():
            if isinstance(value, dict):
                type_map[key] = self._create_type_map_(value)
            else:
                type_map[key] = type(value)

        return type_map

    def _check_structure_(self, type_map: Dict, payload: Dict) -> bool:
        """
        Recursively check if the payload matches the sample structure.

        Parameters
        ----------
        type_map : Dict
            The type map dictionary to use as a reference for structure and keys.

        payload : Dict
            The payload dictionary to verify.

        Returns
        -------
        bool
            True if the payload matches the sample structure, False otherwise.
        """
        for key in type_map:
            if key not in payload:
                return False

            if isinstance(type_map[key], dict):
                if not isinstance(payload[key], dict):
                    return False

                if not self._check_structure_(type_map[key], payload[key]):
                    return False

            else:
                if self.strict and not isinstance(payload[key], type_map[key]):
                    return False
                    
        return True