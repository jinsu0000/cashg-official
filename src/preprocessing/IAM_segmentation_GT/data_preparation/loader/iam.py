import tarfile
from pathlib import Path
from typing import IO, List, Optional, Union

from lxml import etree
from tqdm import tqdm

from drawing.draw import Drawing, Point, Stroke


def load_strokes(file: Union[str, IO[bytes]]) -> Drawing:
    tree = etree.parse(file)

    strokes: List[Stroke] = []
    for stroke in tree.iter("Stroke"):
        stroke_index: int = len(strokes)

        points: List[Point] = []
        for point in stroke.iter("Point"):
            points.append(
                Point(
                    x=int(point.get("x")),
                    y=int(point.get("y")),
                    index=len(points),
                    stroke=stroke_index,
                    time=float(point.get("time")),
                )
            )

        strokes.append(
            Stroke(
                points=points,
                index=stroke_index,
                color=stroke.get("colour"),
                start_time=float(stroke.get("start_time")),
                end_time=float(stroke.get("end_time")),
            )
        )

    return Drawing(strokes)


def get_all_drawings(file: Union[Path, str], id: Optional[str] = None) -> List[Drawing]:

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tqdm(
            tar.getmembers(),
            desc="Loading IAM drawings",
            leave=False,
            dynamic_ncols=True,
        ):
            if member.isfile():
                full_id = member.name[
                    member.name.rfind("/") + 1 : member.name.find(".")
                ]

                if id is not None and id != full_id:
                    continue

                sample = member.name[
                    member.name.find("/", member.name.find("/") + 1)
                    + 1 : member.name.rfind("/")
                ]

                f = tar.extractfile(member)
                if f is None:
                    raise IOError("Could not extract tar file")
                drawing = load_strokes(f)
                drawing.set_sample_name(sample)
                drawing.set_id(full_id)
                drawing.key = f"{drawing.id}.xml"

                samples.append(drawing)

    return samples


def get_sample_names(file: str) -> List[str]:

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isdir():
                if member.name.count("/") == 2:
                    samples.append(member.name[member.name.rfind("/") + 1 :])

    return samples


def get_line_names(file: str) -> List[str]:

    samples = []

    with tarfile.open(file, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                samples.append(
                    member.name[member.name.rfind("/") + 1 : member.name.find(".")]
                )

    return samples
