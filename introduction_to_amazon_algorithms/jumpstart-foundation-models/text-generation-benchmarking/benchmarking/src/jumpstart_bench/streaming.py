import datetime
import io
import json
from typing import Any, Dict, Tuple


class StreamScanner:
    """
    A helper class for parsing the InvokeEndpointWithResponseStream event stream.

    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" challenging"]}\n'
    b'{"outputs": [" problem"]}\n'
    ...
    ```

    While usually each PayloadPart event from the event stream will contain a byte array
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    ```
    {'PayloadPart': {'Bytes': b'{"outputs": '}}
    {'PayloadPart': {'Bytes': b'[" problem"]}\n'}}
    ```

    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\n' character) within
    the buffer via the 'readlines' function. It maintains the position of the last read
    position to ensure that previous bytes are not exposed again.
    """

    def __init__(self) -> None:
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content: bytes) -> None:
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def readlines(self) -> bytes:
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b"\n":
                self.read_pos += len(line) - 1
                yield line[:-1]


def process_response_stream(response: Dict[str, Any]) -> Tuple[str, int]:
    """Scan, parse, and load the JSON response event stream."""
    event_stream = response["Body"]
    scanner = StreamScanner()
    time_utc_first_token = None
    result = b""
    for event in event_stream:
        if time_utc_first_token is None:
            time_utc_first_token = datetime.datetime.utcnow()
        scanner.write(event["PayloadPart"]["Bytes"])
        for line in scanner.readlines():
            result += line
    result = json.loads(result + b"}")
    return result, time_utc_first_token
