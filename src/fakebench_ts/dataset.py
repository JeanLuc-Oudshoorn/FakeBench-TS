"""Generate, save and load the benchmark's random walks."""

import hashlib
import json
import math
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .distributions import FAMILIES

METADATA_KEY = b"fakebench_ts"


@dataclass(frozen=True)
class Spec:
    """Everything that determines a dataset. Changing any field changes the data."""

    name: str = "v1"
    seed: int = 20261120
    horizon: int = 64
    context_lengths: tuple = (128, 512, 2048)
    # Series per context length for a family of weight 1.
    series_per_weight: int = 50
    families: dict = field(
        default_factory=lambda: {
            "normal": 1,
            "student_t": 2,
            "student_t_heavy": 2,
            "laplace": 1,
            "logistic": 1,
            "uniform": 1,
            "rademacher": 1,
            "jump_mixture": 1,
            "skew_gamma": 2,
            "skew_two_point": 2,
        }
    )
    # The size of a step is 10 ** uniform(low, high).
    log10_scale: tuple = (-2.0, 3.0)
    # Half the walks start at zero; the others start this many step sizes
    # (standard deviation) away from it.
    offset_probability: float = 0.5
    offset_sd: float = 100.0

    def __post_init__(self):
        unknown = sorted(set(self.families) - set(FAMILIES))
        if unknown:
            raise ValueError(f"Unknown families {unknown}; known: {sorted(FAMILIES)}")
        if self.horizon < 2 or min(self.context_lengths) < 3:
            raise ValueError("Need a horizon of at least 2 and contexts of at least 3")

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("context_lengths", "log10_scale"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self):
        values = asdict(self)
        values["context_lengths"] = list(self.context_lengths)
        values["log10_scale"] = list(self.log10_scale)
        return values


SPECS = {"v1": Spec()}
REFERENCE_FOLDER = Path(__file__).parent / "reference"
# Windows and Linux round a few logarithms differently, which moves some
# values by about 1e-13 of a step. Anything beyond this is different data.
REFERENCE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Series:
    series_id: str
    track: str
    family: str
    context: np.ndarray
    future: np.ndarray
    # Median of (future value - last context value) for each forecast step:
    # the forecast that minimises expected absolute error is last + this.
    optimal_offset: np.ndarray
    scale: float
    start: float
    params: dict

    @property
    def context_length(self):
        return len(self.context)

    @property
    def horizon(self):
        return len(self.future)

    @property
    def optimal_forecast(self):
        return self.context[-1] + self.optimal_offset


@dataclass(frozen=True)
class Dataset:
    spec: dict
    series: list
    sha256: str

    def __len__(self):
        return len(self.series)

    def __iter__(self):
        return iter(self.series)

    @property
    def horizon(self):
        return self.spec["horizon"]


def generate_series(spec, family_name, context_length, replicate):
    """One walk. Its values depend only on the seed and the three arguments."""
    family = FAMILIES[family_name]
    seed = np.random.SeedSequence(
        spec.seed,
        spawn_key=(zlib.crc32(family_name.encode()), context_length, replicate),
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    params = family.sample_params(rng)
    scale = 10.0 ** rng.uniform(*spec.log10_scale)
    offset = rng.standard_normal() * spec.offset_sd
    start = scale * offset if rng.random() < spec.offset_probability else 0.0
    steps = scale * family.draw(rng, params, context_length + spec.horizon - 1)
    values = start + np.concatenate([[0.0], np.cumsum(steps)])
    return Series(
        series_id=f"{family_name}/c{context_length:04d}/{replicate:04d}",
        track=family.track,
        family=family_name,
        context=values[:context_length],
        future=values[context_length:],
        optimal_offset=scale * family.median_of_sum(params, spec.horizon),
        scale=scale,
        start=start,
        params=params,
    )


def checksum(series):
    digest = hashlib.sha256()
    for item in series:
        digest.update(item.series_id.encode())
        for values in (item.context, item.future, item.optimal_offset):
            digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def generate_dataset(spec=None):
    spec = SPECS["v1"] if spec is None else spec
    series = [
        generate_series(spec, family, context_length, replicate)
        for family, weight in spec.families.items()
        for context_length in spec.context_lengths
        for replicate in range(math.ceil(weight * spec.series_per_weight))
    ]
    return Dataset(spec.to_dict(), series, checksum(series))


def _end_values(dataset):
    """Each walk's last value and last optimal offset: they move if any step does."""
    return {
        "series_id": [s.series_id for s in dataset],
        "last_value": [float(s.future[-1]) for s in dataset],
        "last_optimal_offset": [float(s.optimal_offset[-1]) for s in dataset],
    }


def write_reference(dataset):
    """Record the data a built-in spec produces (maintainers, when adding a spec)."""
    REFERENCE_FOLDER.mkdir(exist_ok=True)
    path = REFERENCE_FOLDER / f"{dataset.spec['name']}.json"
    path.write_text(json.dumps(_end_values(dataset)), encoding="utf-8")
    return path


def matches_reference(dataset):
    """Whether this is the data the built-in spec of that name is known to produce.

    NumPy does not promise the same random numbers in every version, so a
    built-in spec ships with the end values it produced when it was defined.
    Returns None for a spec without a reference (any custom spec).
    """
    name = dataset.spec["name"]
    path = REFERENCE_FOLDER / f"{name}.json"
    if name not in SPECS or dataset.spec != SPECS[name].to_dict() or not path.is_file():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    found = _end_values(dataset)
    if found["series_id"] != reference["series_id"]:
        return False
    tolerance = REFERENCE_TOLERANCE * np.array([s.scale for s in dataset])
    return all(
        bool(np.all(np.abs(np.subtract(found[key], reference[key])) <= tolerance))
        for key in ("last_value", "last_optimal_offset")
    )


def save_dataset(dataset, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    floats = pa.list_(pa.float64())
    table = pa.table(
        {
            "series_id": [s.series_id for s in dataset],
            "track": [s.track for s in dataset],
            "family": [s.family for s in dataset],
            "context_length": pa.array([s.context_length for s in dataset], pa.int32()),
            "scale": [s.scale for s in dataset],
            "start": [s.start for s in dataset],
            "params": [json.dumps(s.params) for s in dataset],
            "context": pa.array([s.context for s in dataset], floats),
            "future": pa.array([s.future for s in dataset], floats),
            "optimal_offset": pa.array([s.optimal_offset for s in dataset], floats),
        }
    )
    metadata = json.dumps({"spec": dataset.spec, "sha256": dataset.sha256})
    pq.write_table(table.replace_schema_metadata({METADATA_KEY: metadata}), path)
    return path


def _float_lists(column):
    """A list<double> column as one NumPy array per row, without Python floats."""
    arrays = []
    for chunk in column.chunks:
        values = chunk.values.to_numpy(zero_copy_only=False)
        offsets = chunk.offsets.to_numpy()
        arrays.extend(
            values[offsets[i] : offsets[i + 1]].copy() for i in range(len(chunk))
        )
    return arrays


def load_dataset(path):
    table = pq.read_table(path)
    stored = (table.schema.metadata or {}).get(METADATA_KEY)
    if stored is None:
        raise ValueError(f"{path} is not a FakeBench-TS dataset")
    metadata = json.loads(stored)
    plain = {
        name: table.column(name).to_pylist()
        for name in ("series_id", "track", "family", "scale", "start", "params")
    }
    lists = {
        name: _float_lists(table.column(name))
        for name in ("context", "future", "optimal_offset")
    }
    series = [
        Series(
            series_id=plain["series_id"][i],
            track=plain["track"][i],
            family=plain["family"][i],
            context=lists["context"][i],
            future=lists["future"][i],
            optimal_offset=lists["optimal_offset"][i],
            scale=plain["scale"][i],
            start=plain["start"][i],
            params=json.loads(plain["params"][i]),
        )
        for i in range(table.num_rows)
    ]
    if checksum(series) != metadata["sha256"]:
        raise ValueError(f"{path} does not match its recorded checksum")
    return Dataset(metadata["spec"], series, metadata["sha256"])
