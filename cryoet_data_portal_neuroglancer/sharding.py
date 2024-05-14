# BSD 3-Clause License

# Copyright (c) 2017, Ignacio Tartavull, William Silversmith, and later authors. (cloud-volume)
# All rights reserved.
# Copyright (c) 2020, William Silversmith, Seung Lab (cloud-files)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import copy
import gzip
import json
from collections import defaultdict, namedtuple

import numpy as np
from cloudfiles.interfaces import BytesIO
from tqdm import tqdm

# All this file is adapted from "cloud-volume" sharding.py file
# https://github.com/seung-lab/cloud-volume/blob/master/cloudvolume/datasource/precomputed/sharding.py


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def jsonify(obj, **kwargs):
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


# Adapted from "cloud-file" compress.py
# https://github.com/seung-lab/cloud-files/blob/master/cloudfiles/compression.py
def gzip_compress(content, compresslevel=None):
    if compresslevel is None:
        compresslevel = 9

    stringio = BytesIO()
    gzip_obj = gzip.GzipFile(mode="wb", fileobj=stringio, compresslevel=compresslevel)
    gzip_obj.write(content)
    gzip_obj.close()
    return stringio.getvalue()


def compress(content, method="gzip", compresslevel=None):
    try:
        compress_methods = {
            "": lambda content, compresslevel: content,
            "gzip": gzip_compress,
        }
        return compress_methods[method](content, compresslevel=compresslevel)
    except KeyError:
        raise ValueError(f"Compression method {method} is unknown") from None


ShardLocation = namedtuple("ShardLocation", ("shard_number", "minishard_number", "remainder"))

uint64 = np.uint64


class ShardingSpecification(object):
    def __init__(
        self,
        type,
        preshift_bits,
        hash,
        minishard_bits,
        shard_bits,
        minishard_index_encoding="raw",
        data_encoding="raw",
    ):
        self.type = type
        self.preshift_bits = uint64(preshift_bits)
        self.hash = hash
        self.minishard_bits = uint64(minishard_bits)
        self.shard_bits = uint64(shard_bits)
        self.minishard_index_encoding = minishard_index_encoding
        self.data_encoding = data_encoding

        self.minishard_mask = self.compute_minishard_mask(self.minishard_bits)
        self.shard_mask = self.compute_shard_mask(self.shard_bits, self.minishard_bits)

        self.validate()

    def clone(self):
        return ShardingSpecification.from_dict(self.to_dict())

    def index_length(self):
        return int((2**self.minishard_bits) * 16)

    @property
    def hash(self):
        return self._hash

    @hash.setter
    def hash(self, val):
        if val == "identity":
            self.hashfn = lambda x: uint64(x)
        elif val == "murmurhash3_x86_128":
            #   self.hashfn = lambda x: uint64(mmh3.hash64(uint64(x).tobytes(), x64arch=False)[0])
            raise NotImplementedError("murmurhash3_x86_128 is not yet supported")
        else:
            raise ValueError("hash {} must be either 'identity' or 'murmurhash3_x86_128'".format(val))

        self._hash = val

    @property
    def preshift_bits(self):
        return self._preshift_bits

    @preshift_bits.setter
    def preshift_bits(self, val):
        self._preshift_bits = uint64(val)

    @property
    def shard_bits(self):
        return self._shard_bits

    @shard_bits.setter
    def shard_bits(self, val):
        self._shard_bits = uint64(val)

    @property
    def minishard_bits(self):
        return self._minishard_bits

    @minishard_bits.setter
    def minishard_bits(self, val):
        val = uint64(val)
        self.minishard_mask = self.compute_minishard_mask(val)
        self._minishard_bits = uint64(val)

    def compute_minishard_mask(self, val):
        if val < 0:
            raise ValueError(str(val) + " must be greater or equal to than zero.")
        elif val == 0:
            return uint64(0)

        minishard_mask = uint64(1)
        for _ in range(val - uint64(1)):
            minishard_mask <<= uint64(1)
            minishard_mask |= uint64(1)
        return uint64(minishard_mask)

    def compute_shard_mask(self, shard_bits, minishard_bits):
        ones64 = uint64(0xFFFFFFFFFFFFFFFF)
        movement = uint64(minishard_bits + shard_bits)
        shard_mask = ~((ones64 >> movement) << movement)
        minishard_mask = self.compute_minishard_mask(minishard_bits)
        return shard_mask & (~minishard_mask)

    @classmethod
    def from_json(cls, vals):
        dct = json.loads(vals.decode("utf8"))
        return cls.from_dict(dct)

    def to_json(self):
        return jsonify(self.to_dict())

    @classmethod
    def from_dict(cls, vals):
        vals = copy.deepcopy(vals)
        vals["type"] = vals["@type"]
        del vals["@type"]
        return cls(**vals)

    def to_dict(self):
        return {
            "@type": self.type,
            "preshift_bits": self.preshift_bits,
            "hash": self.hash,
            "minishard_bits": self.minishard_bits,
            "shard_bits": self.shard_bits,
            "minishard_index_encoding": self.minishard_index_encoding,
            "data_encoding": self.data_encoding,
        }

    def compute_shard_location(self, key):
        chunkid = uint64(key) >> uint64(self.preshift_bits)
        chunkid = self.hashfn(chunkid)
        minishard_number = uint64(chunkid & self.minishard_mask)
        shard_number = uint64((chunkid & self.shard_mask) >> uint64(self.minishard_bits))
        shard_number = format(shard_number, "x").zfill(int(np.ceil(self.shard_bits / 4.0)))
        remainder = chunkid >> uint64(self.minishard_bits + self.shard_bits)

        return ShardLocation(shard_number, minishard_number, remainder)

    def synthesize_shards(self, data, data_offset=None, progress=False):
        """
        Given this specification and a comprehensive listing of
        all the items that could be combined into a given shard,
        synthesize the shard files for this set of labels.

        data: { label: binary, ... }

        e.g. { 5: b'...', 7: b'...' }

        data_offset: { label: offset, ... }

        e.g. { 5: 1234, 7: 5678...' }

        Returns: {
          $filename: binary data,
        }
        """
        return synthesize_shard_files(self, data, data_offset, progress)

    def synthesize_shard(self, labels, data_offset=None, progress=False, presorted=False):
        """
        Assemble a shard file from a group of labels that all belong in the same shard.

        Assembles the .shard file like:
        [ shard index; minishards; all minishard indices ]

        label_group:
          If presorted is True:
            { minishardno: { label: binary, ... }, ... }
          If presorted is False:
            { label: binary }
        progress: show progress bars

        Returns: binary representing a shard file
        """
        return synthesize_shard_file(self, labels, data_offset, progress, presorted)

    def validate(self):
        if self.type not in ("neuroglancer_uint64_sharded_v1",):
            raise ValueError("@type ({}) must be 'neuroglancer_uint64_sharded_v1'.".format(self.type))

        if not (64 > self.preshift_bits >= 0):
            raise ValueError("preshift_bits must be a whole number less than 64: {}".format(self.preshift_bits))

        if not (64 >= self.minishard_bits >= 0):
            raise ValueError("minishard_bits must be between 0 and 64 inclusive: {}".format(self.minishard_bits))

        if not (64 >= self.shard_bits >= 0):
            raise ValueError("shard_bits must be between 0 and 64 inclusive: {}".format(self.shard_bits))

        if self.minishard_bits + self.shard_bits > 64:
            raise ValueError(
                "minishard_bits and shard_bits must sum to less than or equal to 64: minishard_bits<{}> + shard_bits<{}> = {}".format(
                    self.minishard_bits,
                    self.shard_bits,
                    self.minishard_bits + self.shard_bits,
                ),
            )

        if self.hash not in ("identity", "murmurhash3_x86_128"):
            raise ValueError("hash {} must be either 'identity' or 'murmurhash3_x86_128'".format(self.hash))

        if self.minishard_index_encoding not in ("raw", "gzip"):
            raise ValueError("minishard_index_encoding only supports values 'raw' or 'gzip'.")

        if self.data_encoding not in ("raw", "gzip"):
            raise ValueError("data_encoding only supports values 'raw' or 'gzip'.")

    def __str__(self):
        return "ShardingSpecification::" + str(self.to_dict())


def synthesize_shard_files(spec, data, data_offset=None, progress=False):
    """
    From a set of data guaranteed to constitute one or more
    complete and comprehensive shards (no partial shards)
    return a set of files ready for upload.

    WARNING: This function is only appropriate for Precomputed
    meshes and skeletons. Use the synthesize_shard_file (singular)
    function to create arbitrarily named and assigned shard files.

    spec: a ShardingSpecification
    data: { label: binary, ... }
    data_offset: { label: offset, ... }

    Returns: { filename: binary, ... }
    """
    shard_groupings = defaultdict(lambda: defaultdict(dict))
    pbar = tqdm(data.items(), desc="Creating Shard Groupings", disable=(not progress))

    for label, binary in pbar:
        loc = spec.compute_shard_location(label)
        shard_groupings[loc.shard_number][loc.minishard_number][label] = binary

    shard_files = {}

    pbar = tqdm(shard_groupings.items(), desc="Synthesizing Shard Files", disable=(not progress))

    for shardno, shardgrp in pbar:
        filename = str(shardno) + ".shard"
        shard_files[filename] = synthesize_shard_file(
            spec,
            shardgrp,
            data_offset,
            progress=(progress > 1),
            presorted=True,
        )

    return shard_files


# NB: This is going to be memory hungry and can be optimized


def synthesize_shard_file(spec, label_group, data_offset=None, progress=False, presorted=False):
    """
    Assemble a shard file from a group of labels that all belong in the same shard.

    Assembles the .shard file like:
    [ shard index; minishards; all minishard indices ]

    spec: ShardingSpecification
    label_group:
      If presorted is True:
        { minishardno: { label: binary, ... }, ... }
      If presorted is False:
        { label: binary }
    data_offset: { label: offset, ... }
    progress: show progress bars

    Returns: binary representing a shard file
    """
    minishardnos = []
    minishard_indicies = []
    minishards = []

    if presorted:
        minishard_mapping = label_group
    else:
        minishard_mapping = defaultdict(dict)
        pbar = tqdm(label_group.items(), disable=(not progress), desc="Assigning Minishards")
        for label, binary in pbar:
            loc = spec.compute_shard_location(label)
            minishard_mapping[loc.minishard_number][label] = binary

    del label_group

    for minishardno, minishardgrp in tqdm(minishard_mapping.items(), desc="Minishard Indices", disable=(not progress)):
        labels = sorted([int(label) for label in minishardgrp])
        if len(labels) == 0:
            continue

        minishard_index = np.zeros((3, len(labels)), dtype=np.uint64, order="C")
        minishard_components = []

        # label and offset are delta encoded
        last_label = 0
        for i, label in enumerate(labels):
            binary = minishardgrp[label]
            if spec.data_encoding != "raw":
                binary = compress(binary, method=spec.data_encoding)

            # delta encoded [label, offset, size]
            minishard_index[0, i] = label - last_label
            if data_offset is None:
                minishard_index[1, i] = 0  # minishard_index[2, i - 1]
                minishard_index[2, i] = len(binary)
            else:
                # add offset of the actual data if it exists
                minishard_index[1, i] = len(binary) - data_offset[label]
                minishard_index[2, i] = data_offset[label]

            minishard_components.append(binary)
            last_label = label
            del minishardgrp[label]

        minishard = b"".join(minishard_components)
        minishardnos.append(minishardno)
        minishard_indicies.append(minishard_index)
        minishards.append(minishard)

    del minishard_mapping

    cum_minishard_size = 0
    for idx, minishard in zip(minishard_indicies, minishards, strict=False):
        idx[1, 0] += cum_minishard_size
        cum_minishard_size += len(minishard)

    if progress:
        print("Partial assembly of minishard indicies and data... ", end="", flush=True)

    variable_index_part = [idx.tobytes("C") for idx in minishard_indicies]
    if spec.minishard_index_encoding != "raw":
        variable_index_part = [compress(idx, method=spec.minishard_index_encoding) for idx in variable_index_part]

    data_part = b"".join(minishards)
    del minishards

    if progress:
        print("Assembled.")

    fixed_index = np.zeros((int(2**spec.minishard_bits), 2), dtype=np.uint64, order="C")

    start = len(data_part)
    end = len(data_part)
    for i, idx in zip(minishardnos, variable_index_part, strict=False):
        start = end
        end += len(idx)
        fixed_index[i, 0] = start
        fixed_index[i, 1] = end

    if progress:
        print("Final assembly... ", end="", flush=True)

    # The order here is important. The fixed index must go first because the locations
    # of the other parts are calculated with it implicitly in front. The variable
    # index must go last because otherwise compressing it will affect offset of the
    # data it is attempting to index.

    result = fixed_index.tobytes("C") + data_part + b"".join(variable_index_part)

    if progress:
        print("Done.")

    return result
