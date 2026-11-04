from collections.abc import Mapping

RecordInput = (
	dict[str, str]  # mapping record_id -> text
	| list[tuple[str, str]]  # list of (record_id, text)
	| list[Mapping[str, str]]  # list of {"id": "text"} (must be single entry)
)
