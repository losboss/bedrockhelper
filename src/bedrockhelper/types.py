from typing import Dict, List, Mapping, Tuple, TypedDict, Union

from typing_extensions import NotRequired


RecordInput = Union[
	Dict[str, str],  # mapping record_id -> text
	List[Tuple[str, str]],  # list of (record_id, text)
	List[Mapping[str, str]],  # list of {"id": "text"} (must be single entry)
]


class RetryDict(TypedDict):
	max_attempts: NotRequired[int]
	mode: NotRequired[str]
