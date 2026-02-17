from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cmp_to_key
import math
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET


class Align(Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class ColDataType(Enum):
    UNKNOWN = "unknown"
    NUMERIC = "Numeric"
    STRING = "String"


RowFilter = Callable[["TableOfData", int], bool]


class ColumnInfo:
    def __init__(self, hdr: str, humanform: str | None = None) -> None:
        self.datatype = ColDataType.UNKNOWN
        self.alignHeader = Align.CENTER
        self.alignData = Align.RIGHT
        self.Name = ""
        self.HumanForm = ""
        self.Evaluate: Any = None
        self.Original = hdr
        self.format = ""
        self.maxWidth = 0
        self.direction = 0
        self.Hidden = False
        if hdr == "":
            if humanform is not None:
                self.HumanForm = humanform
            return

        parts = hdr.split("|")
        self.Name = parts[0]
        self.HumanForm = parts[0]

        if len(parts) > 1:
            align_spec = self._correct(parts[1])
            for c in align_spec:
                if c == "L":
                    self.alignHeader = Align.LEFT
                elif c == "C":
                    self.alignHeader = Align.CENTER
                elif c == "R":
                    self.alignHeader = Align.RIGHT
                elif c == "l":
                    self.alignData = Align.LEFT
                elif c == "c":
                    self.alignData = Align.CENTER
                elif c == "r":
                    self.alignData = Align.RIGHT
                elif c == "+":
                    self.direction = -1
                elif c == "-":
                    self.direction = 1
                elif c in ("h", "H"):
                    self.Hidden = True

        if len(parts) > 2:
            self.format = parts[2].strip()

        if len(parts) > 3:
            self.Evaluate = parts[3].strip()

        if humanform is not None:
            self.HumanForm = humanform

    @property
    def Alignment(self) -> str:
        ah = "C"
        if self.alignHeader == Align.RIGHT:
            ah = "R"
        elif self.alignHeader == Align.LEFT:
            ah = "L"

        ad = "c"
        if self.alignData == Align.RIGHT:
            ad = "r"
        elif self.alignData == Align.LEFT:
            ad = "l"

        res = ah + ad
        if self.direction == -1:
            res += "+"
        elif self.direction == 1:
            res += "-"
        return res

    @Alignment.setter
    def Alignment(self, value: str) -> None:
        self.alignHeader = Align.CENTER
        if "R" in value:
            self.alignHeader = Align.RIGHT
        if "L" in value:
            self.alignHeader = Align.LEFT

        self.alignData = Align.CENTER
        if "r" in value:
            self.alignData = Align.RIGHT
        if "l" in value:
            self.alignData = Align.LEFT

        if "+" in value:
            self.direction = -1
        elif "-" in value:
            self.direction = 1
        else:
            self.direction = 0

    @property
    def Format(self) -> str:
        return self.format

    @Format.setter
    def Format(self, value: str) -> None:
        self.format = value

    @staticmethod
    def _correct(a: str) -> str:
        if len(a) >= 2:
            c0 = a[0]
            c1 = a[1]
            lower = {"l", "c", "r"}
            if c0 in lower and c1 in lower:
                return c1.upper() + a[1:]
        return a


class Columns(list[ColumnInfo]):
    maxCircle = 10

    def __init__(self, *columns: str | ColumnInfo) -> None:
        super().__init__()
        self.icircle = 0
        self.colsHash: dict[str, int] | None = None
        for hdr in columns:
            if isinstance(hdr, ColumnInfo):
                self.append(hdr)
            else:
                self.AddColumn(hdr)

    def append(self, item: ColumnInfo) -> None:  # type: ignore[override]
        super().append(item)
        self.colsHash = None
        if len(self) == 1:
            self.icircle = 0
        else:
            self.icircle = min(self.icircle, len(self) - 1)

    def AddColumn(self, hdr: str, humanform: str | None = None) -> None:
        self.append(ColumnInfo(hdr, humanform))

    def index_of(self, col: str) -> int:
        if not self:
            return -1
        if 0 <= self.icircle < len(self) and col == self[self.icircle].Name:
            return self.icircle

        if len(self) < self.maxCircle:
            start = self.icircle
            while True:
                self.icircle = (self.icircle + 1) % len(self)
                if col == self[self.icircle].Name:
                    return self.icircle
                if self.icircle == start:
                    return -1
        else:
            if self.colsHash is None:
                self.colsHash = {self[i].Name: i for i in range(len(self))}
            found = self.colsHash.get(col)
            if found is None:
                return -1
            self.icircle = found
            return found

    def __getitem__(self, key: int | str) -> ColumnInfo | int:  # type: ignore[override]
        if isinstance(key, str):
            return self.index_of(key)
        return super().__getitem__(key)

    def ColumnsLike(self, frag: str) -> "Columns":
        res = Columns()
        frag_l = frag.lower()
        for ci in self:
            if frag == "*" or frag_l in ci.Name.lower():
                res.append(ci)
        return res


class TableOfData:
    def __init__(self, *allcols: str, fname: str | None = None, csv: bool = False, addCols: list[str] | None = None) -> None:
        self.Cols = Columns()
        self.Rows: list[list[Any]] = []
        self.keyCol = -1
        self.keyHash: dict[str, int] | None = None

        if allcols:
            for col in allcols:
                self.Cols.AddColumn(col)

        if fname is not None:
            self.LoadTabDelimited(fname, csv, *(addCols or []))

    @property
    def Count(self) -> int:
        return len(self.Rows)

    def AddColumn(self, col: str, humanform: str | None = None) -> None:
        if self.Rows:
            raise Exception("TableOfData can't add columns after data has been stored in the table")
        self.Cols.AddColumn(col, humanform)

    def Rankable(self) -> bool:
        return any(ci.direction != 0 for ci in self.Cols)

    def RankOn(self) -> str:
        chunks: list[str] = []
        for ci in self.Cols:
            if ci.direction != 0:
                chunks.append(f"{ci.Name}{'-' if ci.direction > 0 else '+'}")
        return " ".join(chunks)

    def AddRow(self) -> list[Any]:
        newrow: list[Any] = [None] * len(self.Cols)
        self.Rows.append(newrow)
        return newrow

    def KeyColumn(self, col: str) -> None:
        self.keyCol = self.Cols.index_of(col)
        if self.keyCol >= 0:
            self.keyHash = {}
            for i, row in enumerate(self.Rows):
                key = self.Text(i, self.keyCol)
                if key:
                    self.keyHash[key] = i

    def RowIndex(self, key: str) -> int:
        if self.keyHash is None:
            return -1
        return self.keyHash.get(key, -1)

    def Row(self, selector: int | str) -> list[Any] | None:
        if isinstance(selector, int):
            return self.Rows[selector]
        irow = self.RowIndex(selector)
        if irow < 0:
            return None
        return self.Rows[irow]

    def Key(self, irow: int) -> str:
        if self.keyCol < 0:
            return ""
        return self.Text(irow, self.keyCol)

    def _set_key(self, irow: int, key: Any) -> None:
        if self.keyHash is None:
            return
        if not isinstance(key, str) or key == "":
            raise Exception("TableOfData: key must be a non-null string")
        if self.RowIndex(key) >= 0:
            raise Exception(f'TableOfData: key "{key}" is not unique')
        row = self.Rows[irow]
        existing = row[self.keyCol]
        if existing is not None:
            raise Exception("TableOfData: cannot replace or overwrite existing key")
        row[self.keyCol] = key
        self.keyHash[key] = irow

    def _col_index(self, scol_or_icol: int | str) -> int:
        if isinstance(scol_or_icol, int):
            return scol_or_icol
        return self.Cols.index_of(scol_or_icol)

    def get_cell(self, row_sel: int | str, col_sel: int | str) -> Any:
        icol = self._col_index(col_sel)
        if isinstance(row_sel, int):
            return self.Rows[row_sel][icol]
        if self.keyHash is None:
            return None
        krow = self.keyHash.get(row_sel)
        if krow is None:
            return None
        return self.Rows[krow][icol]

    def set_cell(self, row_sel: int | str, col_sel: int | str, value: Any) -> None:
        icol = self._col_index(col_sel)
        if isinstance(row_sel, int):
            row = self.Rows[row_sel]
            if icol == self.keyCol:
                self._set_key(row_sel, value)
            else:
                row[icol] = value
            return

        if self.keyHash is None or row_sel == "":
            return

        krow = self.keyHash.get(row_sel)
        if krow is None:
            irow = len(self.Rows)
            row = self.AddRow()
        else:
            irow = krow
            row = self.Rows[irow]

        if icol == self.keyCol:
            self._set_key(irow, value)
        else:
            row[icol] = value

    def __getitem__(self, key: tuple[int | str, int | str]) -> Any:
        row_sel, col_sel = key
        return self.get_cell(row_sel, col_sel)

    def __setitem__(self, key: tuple[int | str, int | str], value: Any) -> None:
        row_sel, col_sel = key
        self.set_cell(row_sel, col_sel, value)

    def Number(self, row_or_irow: list[Any] | int, scol_or_icol: int | str) -> float:
        if isinstance(row_or_irow, int):
            row = self.Rows[row_or_irow]
        else:
            row = row_or_irow
        icol = self._col_index(scol_or_icol)

        cell = row[icol]
        if cell is None:
            return math.nan
        if isinstance(cell, bool):
            return float(cell)
        if isinstance(cell, (int, float)):
            return float(cell)
        if isinstance(cell, str):
            c = cell.strip()
            if c == "" or c == "N/A":
                return math.nan
            try:
                num = float(c)
                row[icol] = num
                return num
            except ValueError:
                return math.nan
        return math.nan

    def Text(self, row_or_irow: list[Any] | int, scol_or_icol: int | str) -> str:
        if isinstance(row_or_irow, int):
            row = self.Rows[row_or_irow]
        else:
            row = row_or_irow
        icol = self._col_index(scol_or_icol)
        cell = row[icol]
        return cell if isinstance(cell, str) else ""

    @staticmethod
    def ReadLines(filename: str) -> list[str]:
        path = Path(filename)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def LoadTabDelimited(self, fname: str, csv: bool, *addCols: str) -> None:
        separator = "," if csv else "\t"
        fpath = fname.lower()
        if not fpath.endswith(".txt"):
            fpath = f"{fpath}.txt"
        path = Path(fpath)
        if not path.exists():
            raise Exception(f'file "{fpath}" not found')

        lines = self.ReadLines(fpath)
        hdr_parts = lines[0].split(separator)
        for hp in hdr_parts:
            self.AddColumn(hp)
        for extra in addCols:
            self.AddColumn(extra)

        for k in range(1, len(lines)):
            parts = lines[k].lower().split(separator)
            newrow = self.AddRow()
            for i, part in enumerate(parts):
                newrow[i] = None if part == "" else part

    def _format_value(self, value: Any, fmt: str) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if fmt:
            return format(value, fmt)
        return str(value)

    def SaveTabDelimited(self, filename: str, topPct: float) -> None:
        nr = float(self.Count)
        if nr == 0:
            return

        if topPct <= 2.0:
            topPct = 2.0
        if topPct > 100:
            topPct = 100.0

        nr = round(nr * topPct / 100.0)
        nr = min(nr, self.Count)
        if nr < 10:
            nr = min(self.Count, 10)

        lines: list[str] = []
        line = ""
        for ci in self.Cols:
            if line:
                line += "\t"
            line += ci.Name
        lines.append(line)

        remaining = int(nr)
        for row in self.Rows:
            line = ""
            for icol, cell in enumerate(row):
                if self.Cols[icol].Hidden:
                    continue
                cstring = self._format_value(cell, self.Cols[icol].format)
                if line:
                    line += "\t"
                if cstring:
                    line += cstring
            lines.append(line)
            remaining -= 1
            if remaining < 0:
                break

        path = Path(filename)
        if path.exists():
            path.unlink()
        with path.open("w", encoding="utf-8") as f:
            for aline in lines:
                f.write(aline)
                f.write("\n")

    def SaveXML(self, filename: str, topPct: float) -> None:
        nr = float(self.Count)
        if nr == 0:
            return

        if topPct <= 2.0:
            topPct = 2.0
        if topPct > 100:
            topPct = 100.0

        nr = round(nr * topPct / 100.0)
        nr = min(nr, self.Count)
        if nr < 10:
            nr = min(self.Count, 10)

        report = ET.Element("Report")
        remaining = int(nr)
        for row in self.Rows:
            xrow = ET.SubElement(report, "Row")
            for icol, ci in enumerate(self.Cols):
                if ci.Hidden:
                    continue
                cell = row[icol]
                cstring = self._format_value(cell, ci.format)
                xcell = ET.SubElement(xrow, ci.Name)
                xcell.text = cstring
            remaining -= 1
            if remaining < 0:
                break

        tree = ET.ElementTree(report)
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    def NumericColumn(self, ic: int, defValue: float) -> bool:
        if ic < 0 or ic >= len(self.Cols):
            return False

        res = True
        converted: list[float] = [defValue] * len(self.Rows)
        for irow, row in enumerate(self.Rows):
            c = row[ic]
            if c is None:
                continue
            if isinstance(c, bool):
                converted[irow] = float(c)
                continue
            if isinstance(c, (int, float)):
                converted[irow] = float(c)
                continue
            if not isinstance(c, str):
                res = False
                continue
            s = c.lower()
            if s in ("nan", "n/a"):
                continue
            try:
                converted[irow] = float(s)
            except ValueError:
                res = False

        cinfo: ColumnInfo = self.Cols[ic]  # type: ignore[assignment]
        if res:
            cinfo.datatype = ColDataType.NUMERIC
            if cinfo.Format == "":
                cinfo.Format = ".2f"
            for irow, row in enumerate(self.Rows):
                row[ic] = converted[irow]
        else:
            cinfo.datatype = ColDataType.STRING
            cinfo.Alignment = "Cl"
            for row in self.Rows:
                x = row[ic]
                if isinstance(x, str) and len(x) > 30:
                    row[ic] = x[:30]
        return res

    def RemoveNArows(self, ic: int) -> None:
        nona: list[list[Any]] = []
        for row in self.Rows:
            s = row[ic]
            if isinstance(s, str) and s.lower() != "n/a":
                nona.append(row)
        self.Rows = nona

    def CleanUp(self, fixnumbers: bool, nona: list[str] | None) -> None:
        if nona is not None:
            for keyname in nona:
                keycol = -1
                for ic, col in enumerate(self.Cols):
                    if col.Name == keyname:
                        keycol = ic
                        break
                if keycol >= 0:
                    self.RemoveNArows(keycol)

        if fixnumbers:
            for icol in range(len(self.Cols)):
                self.NumericColumn(icol, math.nan)

    def Subtable(self, selector: RowFilter | None, *subcols: str) -> "TableOfData":
        if not subcols:
            allsubtablecols = [c.Original for c in self.Cols]
        elif subcols[0] == "*":
            allsubtablecols = [c.Original for c in self.Cols] + list(subcols[1:])
        else:
            allsubtablecols = list(subcols)

        subtable = TableOfData(*allsubtablecols)
        cindex = [self.Cols.index_of(c.Name) for c in subtable.Cols]

        for irow, row in enumerate(self.Rows):
            if selector is None or selector(self, irow):
                subrow = subtable.AddRow()
                for j, cix in enumerate(cindex):
                    if cix >= 0:
                        subrow[j] = row[cix]

        if self.keyCol >= 0:
            kcol = self.Cols[self.keyCol].Name
            subtable.KeyColumn(kcol)
        return subtable

    def TableRanker(self, *criteria: str) -> "GeneralRanker":
        ranker = GeneralRanker(*criteria)
        reject: list[DatumToRank] = []

        for irow, row in enumerate(self.Rows):
            rankrow = [0.0] * (ranker.nCriteria + 1)
            skip = False
            for icol in range(ranker.nCriteria):
                crit = ranker.Criteria[icol].Name
                ix = self.Cols.index_of(crit)
                if ix >= 0:
                    num = self.Number(irow, crit)
                    if math.isnan(num):
                        skip = True
                    else:
                        rankrow[icol] = num
            key = row[self.keyCol] if self.keyCol >= 0 and isinstance(row[self.keyCol], str) else ""
            dr = DatumToRank(key, irow, *rankrow)
            if skip:
                reject.append(dr)
            else:
                ranker.Add(dr)

        ranker.rejects = reject
        return ranker

    def Ranked(self, ranked: list["DatumToRank"]) -> "TableOfData":
        res = TableOfData()
        res.Cols = self.Cols
        for rd in ranked:
            irow = int(rd.BackPointer)
            row = self.Rows[irow]
            newrow = res.AddRow()
            for k in range(len(newrow)):
                newrow[k] = row[k]

        if self.keyCol >= 0:
            res.KeyColumn(self.Cols[self.keyCol].Name)
        return res

    def __str__(self) -> str:
        colsToShow = Columns()
        for ci in self.Cols:
            if not ci.Hidden:
                colsToShow.append(ci)

        wholeRow = len(self.Cols) == len(colsToShow)
        tt = TextTable(colsToShow)
        for row in self.Rows:
            if wholeRow:
                tt.NewLine(*row)
            else:
                for ic, ci in enumerate(self.Cols):
                    if ci.Hidden:
                        continue
                    if ci.format != "":
                        tt.NextCell(self.Number(row, ic))
                    else:
                        tt.NextCell(row[ic])
        return tt.Print()


@dataclass
class DatumToRank:
    Name: str
    BackPointer: Any
    Values: list[float]
    ColumnRank: list[float]

    def __init__(self, name: str, backpointer: Any, *values: float) -> None:
        self.Name = name
        self.BackPointer = backpointer
        self.Values = [0.0] * (len(values) + 1)
        for i, v in enumerate(values):
            self.Values[i] = v
        self.ColumnRank = [0.0] * (len(values) + 1)


class GeneralRanker:
    def __init__(self, *criteria: str | TableOfData) -> None:
        self.Criteria = Columns()
        self.nCriteria = 0
        self.Data: list[DatumToRank] = []
        self.rejects: list[DatumToRank] = []

        if len(criteria) == 1 and isinstance(criteria[0], TableOfData):
            self._init_from_table(criteria[0])
        else:
            crit = [c for c in criteria if isinstance(c, str)]
            self.Criteria = Columns(*crit)
            self.nCriteria = len(self.Criteria)
            self.Criteria.AddColumn("Rank|+")
            self.Data = []

    def _init_from_table(self, table: TableOfData) -> None:
        self.Criteria = Columns()
        for ci in table.Cols:
            if ci.direction != 0:
                self.Criteria.AddColumn(ci.Original)

        self.nCriteria = len(self.Criteria)
        if self.nCriteria == 0:
            raise Exception("GeneralRanker constructor: table contains no rankable columns")

        self.Criteria.AddColumn("Rank|+")
        self.Data = []
        self.rejects = []
        for irow in range(table.Count):
            values = [0.0] * (self.nCriteria + 1)
            skip = False
            for i in range(self.nCriteria):
                ci = self.Criteria[i]
                v = table.Number(irow, ci.Name)
                if math.isnan(v) or math.isinf(v):
                    skip = True
                values[i] = v
            dr = DatumToRank(table.Key(irow), irow, *values)
            if skip:
                self.rejects.append(dr)
            else:
                self.Add(dr)

    def Add(self, datum: DatumToRank) -> None:
        self.Data.append(datum)

    def _assign_ranks(self, criterion: int) -> None:
        prevK = 0
        sumK = 0.0
        prevV = self.Data[0].Values[criterion]

        for k in range(len(self.Data)):
            v = self.Data[k].Values[criterion]
            if v == prevV:
                sumK += k
            else:
                avgRank = sumK / (k - prevK) + 1
                for kk in range(prevK, k):
                    self.Data[kk].ColumnRank[criterion] = avgRank
                sumK = float(k)
                prevK = k
                prevV = v

        k = len(self.Data)
        avgRank = sumK / (k - prevK) + 1
        for kk in range(prevK, k):
            self.Data[kk].ColumnRank[criterion] = avgRank

    @staticmethod
    def _directional_compare(col: int, direction: int):
        def compare(left: DatumToRank, right: DatumToRank) -> int:
            vL = left.Values[col]
            vR = right.Values[col]
            res = 0
            if vL < vR:
                res = 1
            elif vL > vR:
                res = -1
            return res * direction

        return compare

    def Rank(self, nToReturn: int = 0) -> list[DatumToRank]:
        if not self.Data:
            return []

        for i in range(self.nCriteria):
            self.Data.sort(key=cmp_to_key(self._directional_compare(i, self.Criteria[i].direction)))
            self._assign_ranks(i)

        for datum in self.Data:
            total = 0.0
            for k in range(self.nCriteria):
                total += datum.ColumnRank[k]
            datum.Values[self.nCriteria] = total

        self.Data.sort(key=cmp_to_key(self._directional_compare(self.nCriteria, -1)))
        self._assign_ranks(self.nCriteria)

        if nToReturn <= 0:
            nToReturn = len(self.Data)
        else:
            nToReturn = min(nToReturn, len(self.Data))
        return [self.Data[i] for i in range(nToReturn)]


class TextTable:
    class Cell:
        def __init__(self, c: Any, fmt: str) -> None:
            self.obj = c
            if self.obj is None:
                self.disp = ""
            elif isinstance(self.obj, str):
                self.disp = self.obj
            elif isinstance(self.obj, datetime):
                self.disp = f"{self.obj.month}/{self.obj.day}/{self.obj.year}"
            elif fmt:
                self.disp = format(self.obj, fmt)
            else:
                self.disp = str(self.obj)

    def __init__(self, *header: str | Columns) -> None:
        if len(header) == 1 and isinstance(header[0], Columns):
            self.cols = header[0]
        else:
            string_header = [h for h in header if isinstance(h, str)]
            self.cols = Columns(*string_header)

        self.numCols = len(self.cols)
        self.nextCol = 0
        self.rows: list[list[TextTable.Cell | None]] = []
        self.colSpace = 2

    def NewLine(self, *data: Any) -> None:
        row: list[TextTable.Cell | None] = [None] * self.numCols
        cc = min(len(data), self.numCols)
        for k in range(cc):
            row[k] = TextTable.Cell(data[k], self.cols[k].format)
        self.rows.append(row)
        self.nextCol = cc

    def NextCell(self, datum: Any) -> None:
        if self.nextCol >= self.numCols or len(self.rows) == 0:
            self.NewLine()
        row = self.rows[-1]
        row[self.nextCol] = TextTable.Cell(datum, self.cols[self.nextCol].format)
        self.nextCol += 1

    def _justify(self, sep: bool, field: str, align: Align, width: int) -> str:
        l = len(field)
        chunks: list[str] = []
        if sep:
            chunks.append(" " * self.colSpace)

        if l == width:
            chunks.append(field)
        elif align == Align.LEFT:
            chunks.append(field)
            chunks.append(" " * (width - l))
        elif align == Align.RIGHT:
            chunks.append(" " * (width - l))
            chunks.append(field)
        else:
            pad = max(0, width - l)
            rpad = pad // 2
            lpad = pad - rpad
            if lpad > 0:
                chunks.append(" " * lpad)
            chunks.append(field)
            if rpad > 0:
                chunks.append(" " * rpad)

        return "".join(chunks)

    def Print(self) -> str:
        if self.numCols == 0:
            return "\n\n"

        for j in range(self.numCols):
            self.cols[j].maxWidth = len(self.cols[j].Name)

        for row in self.rows:
            for j in range(self.numCols):
                cell = row[j]
                if cell is not None:
                    self.cols[j].maxWidth = max(self.cols[j].maxWidth, len(cell.disp))

        headline = self._justify(False, self.cols[0].Name, self.cols[0].alignHeader, self.cols[0].maxWidth)
        underline = "-" * self.cols[0].maxWidth

        for j in range(1, self.numCols):
            headline += self._justify(True, self.cols[j].Name, self.cols[j].alignHeader, self.cols[j].maxWidth)
            underline += " " * self.colSpace + "-" * self.cols[j].maxWidth

        out: list[str] = ["", headline, underline]

        for row in self.rows:
            line = ""
            for j in range(self.numCols):
                cell = row[j]
                text = "" if cell is None else cell.disp
                line += self._justify(j != 0, text, self.cols[j].alignData, self.cols[j].maxWidth)
            out.append(line)

        out.append(underline)
        if len(self.rows) > 20:
            out.append(headline)
        out.append("")
        return "\n".join(out)
