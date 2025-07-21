import os

from mm_toolbox.logging.advanced.handlers.base import BaseLogHandler

class FileLogHandler(BaseLogHandler):
    """
    A log handler that appends log messages to a text file.
    """

    def __init__(self, filepath: str, create: bool = False):
        """
        Initialize the FileLogHandler with a target file path.

        Args:
            filepath (str): Path to the text file for appending logs. Must end with ".txt".
            create (bool, optional): If True, create the file if it doesn't exist. Defaults to False.

        Raises:
            ValueError: If the provided filepath does not end with ".txt".
        """
        super().__init__()
        if not filepath.endswith(".txt"):
            raise ValueError(f"Invalid filepath; expected string ending with '.txt' but got '{filepath}'")
        self.filepath = filepath

        self.create = create
        if self.create:
            # Create the file if it doesn't exist, or truncate it if it does
            try:
                directory = os.path.dirname(self.filepath)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                
                # Create or truncate the file
                with open(self.filepath, "w") as f:
                    pass
            except Exception as e:
                print(f"Failed to create or truncate file; {e}")

    def push(self, name, logs):
        try:
            with open(self.filepath, "a") as file:
                msgs = "\n".join([
                    self.format_log(
                        name=name, 
                        time_ns=log[0], 
                        level=log[1], 
                        msg=log[2]
                    ) for log in logs
                ])
                file.write(msgs)
                file.flush()

        except Exception as e:
            print(f"Failed to write logs to file; {e}")