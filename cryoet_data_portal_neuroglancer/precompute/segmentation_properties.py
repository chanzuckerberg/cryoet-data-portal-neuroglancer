import json
from pathlib import Path

from cryoet_data_portal_neuroglancer.models.json_generator import SegmentPropertyJSONGenerator


def write_segment_properties(base_folder: str | Path, ids: list[int], labels: list[str]):
    segment_generator = SegmentPropertyJSONGenerator(ids=ids, labels=labels)
    segment_properties = segment_generator.generate_json()
    segment_properties_path = Path(base_folder) / "segment_properties" / "info"
    segment_properties_path.parent.mkdir(exist_ok=True, parents=True)
    segment_properties_path.write_text(json.dumps(segment_properties, indent=2))
