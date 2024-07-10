import json
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from cloudvolume.datasource.precomputed.sharding import ShardingSpecification
from cloudvolume.lib import jsonify
from neuroglancer import CoordinateSpace
from neuroglancer.write_annotations import AnnotationWriter


@dataclass
class AnnotationEncoder:
    """Base class for encoding annotations to Neuroglancer format."""

    output_path: Path
    resolution: float
    coordinate_space: CoordinateSpace
    names_by_id: dict[int, str]
    shard_by_id: Optional[tuple[int, int]] = (0, 10)
    label_key_mapper: Callable[[dict[str, Any]], int] = lambda x: 0
    color_mapper: Callable[[dict[str, Any]], tuple[int, int, int]] = lambda x: (255, 255, 255)
    _writer: AnnotationWriter = field(init=False, default=None)

    def __post_init__(self):
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._prepare_writer_specifications()

    def write_annotations(self):
        self._writer.properties.sort(key=lambda prop: prop.id != "name")
        self._writer.write(self.output_path)
        self._shard()

    @abstractmethod
    def _prepare_writer_specifications(self):
        pass

    @abstractmethod
    def process_data_to_writer_specifications(
        self,
        data: list[dict[str, Any]],
        metadata: dict[str, Any],
    ):
        pass

    def _shard(self):
        """Combine all the single annotation files into a set of sharded files.

        This is performed in a post-processing step after all the annotations have been written.

        Parameters
        ----------
        shard_by_id : tuple[int, int], optional
            Tuple of shard_bits and minishard_bits, by default (0, 10)
            If None, no sharding will be done.
        """
        if self.shard_by_id and len(self.shard_by_id) == 2:
            shard_bits, minishard_bits = self.shard_by_id
            self._shard_by_id_index(shard_bits, minishard_bits)

    def _shard_by_id_index(self, shard_bits: int, minishard_bits: int):
        sharding_specification = ShardingSpecification(
            type="neuroglancer_uint64_sharded_v1",
            preshift_bits=0,
            hash="identity",
            minishard_bits=minishard_bits,
            shard_bits=shard_bits,
            minishard_index_encoding="gzip",
            data_encoding="gzip",
        )
        labels = {}
        for file in (self.output_path / "by_id").iterdir():
            if ".shard" not in file.name:
                labels[int(file.name)] = file.read_bytes()
                file.unlink()

        shard_files = sharding_specification.synthesize_shards(labels, progress=True)
        for shard_filename, shard_content in shard_files.items():
            (self.output_path / "by_id" / shard_filename).write_bytes(shard_content)

        info_path = self.output_path / "info"
        info = json.load(info_path.open("r", encoding="utf-8"))
        info["by_id"]["sharding"] = sharding_specification.to_dict()
        info_path.write_text(jsonify(info, indent=2))
