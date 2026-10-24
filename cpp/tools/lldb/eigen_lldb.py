"""LLDB data formatters for Eigen dense matrices, vectors and arrays.

Shows Eigen::Matrix, Eigen::Array (VectorXd, MatrixXd, Vector3d, ...) and
contiguous Eigen::Map views with their values instead of raw storage:

    (Eigen::MatrixXd) m = 2x3 [[1, 2, 3], [4, 5, 6]]
    (Eigen::VectorXd) v = 4 [0.1, 0.2, 0.3, 0.4]

Expanding a variable lists its elements as [i] (vectors) or [i,j] (matrices).

Load it in ~/.lldbinit (CLion and command-line lldb both read that file):

    command script import /path/to/nabia/tools/lldb/eigen_lldb.py

Written for the Eigen 5 storage layout (DenseStorage::m_data / m_rows / m_cols),
which matches Eigen 3.4 for these members.
"""

import re

import lldb

# Limits keep the debugger responsive on large matrices.
SUMMARY_MAX_ROWS = 6
SUMMARY_MAX_COLS = 8
CHILDREN_MAX = 1000

DYNAMIC = -1
ROW_MAJOR_BIT = 0x1

_ARGS = re.compile(r"^(?:const )?Eigen::(?:Matrix|Array)<(.+)>$")


def _template_args(type_name):
    """Splits 'Eigen::Matrix<double, -1, 1, 0, -1, 1>' into its top-level arguments."""
    match = _ARGS.match(type_name)
    if not match:
        return None
    args, depth, current = [], 0, ""
    for ch in match.group(1):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    args.append(current.strip())
    return args


class _Layout:
    """Where the elements of an Eigen object live and how they are ordered."""

    def __init__(self, valobj):
        self.valid = False
        self.error = None
        value = valobj.GetNonSyntheticValue()
        value_type = value.GetType().GetCanonicalType()
        if value_type.IsReferenceType() or value_type.IsPointerType():
            # The dereferenced value comes back with these formatters applied: unwrap again.
            value = value.Dereference().GetNonSyntheticValue()
            value_type = value.GetType().GetCanonicalType()
        value_type = value_type.GetUnqualifiedType()
        name = value_type.GetName()

        is_map = name.startswith("Eigen::Map<")
        plain_type = value_type.GetTemplateArgumentType(0).GetCanonicalType().GetUnqualifiedType() if is_map else value_type
        args = _template_args(plain_type.GetName())
        if not args or len(args) < 4:
            self.error = "unsupported Eigen type"
            return
        self.scalar_type = plain_type.GetTemplateArgumentType(0)
        compile_rows, compile_cols, options = int(args[1]), int(args[2]), int(args[3])
        self.row_major = bool(options & ROW_MAJOR_BIT)
        self.is_vector = compile_rows == 1 or compile_cols == 1

        if is_map:
            # MapBase: PointerType m_data; variable_if_dynamic<Index, N> m_rows, m_cols.
            if not re.search(r"Eigen::Stride<0, 0>", name):
                self.error = "strided Map (not shown)"
                return
            self.address = value.GetChildMemberWithName("m_data").GetValueAsUnsigned()
            self.rows = compile_rows if compile_rows != DYNAMIC else _dynamic_index(value, "m_rows")
            self.cols = compile_cols if compile_cols != DYNAMIC else _dynamic_index(value, "m_cols")
        else:
            storage = value.GetChildMemberWithName("m_storage")
            data = storage.GetChildMemberWithName("m_data")
            if data.GetType().IsPointerType():          # heap storage (dynamic size)
                self.address = data.GetValueAsUnsigned()
            else:                                       # inline plain_array (fixed size)
                self.address = data.GetChildMemberWithName("array").GetLoadAddress()
            self.rows = compile_rows if compile_rows != DYNAMIC else storage.GetChildMemberWithName("m_rows").GetValueAsUnsigned()
            self.cols = compile_cols if compile_cols != DYNAMIC else storage.GetChildMemberWithName("m_cols").GetValueAsUnsigned()

        self.size = self.rows * self.cols
        self.scalar_size = self.scalar_type.GetByteSize()
        self.valid = self.size == 0 or self.address != 0

    def offset(self, i, j):
        """Byte offset of element (i, j)."""
        index = i * self.cols + j if self.row_major else j * self.rows + i
        return index * self.scalar_size

    def element(self, parent, name, i, j):
        return parent.CreateValueFromAddress(name, self.address + self.offset(i, j), self.scalar_type)


def _dynamic_index(value, member):
    """Value of a variable_if_dynamic<Index, Dynamic> member (stored in m_value)."""
    holder = value.GetChildMemberWithName(member)
    inner = holder.GetChildMemberWithName("m_value")
    return (inner if inner.IsValid() else holder).GetValueAsUnsigned()


def _format(value):
    """Shortest text that reads back as the same number (0.1, not 0.10000000000000001)."""
    basic = value.GetType().GetCanonicalType().GetBasicType()
    if basic in (lldb.eBasicTypeDouble, lldb.eBasicTypeFloat):
        error = lldb.SBError()
        data = value.GetData()
        x = data.GetDouble(error, 0) if basic == lldb.eBasicTypeDouble else data.GetFloat(error, 0)
        if error.Success():
            text = repr(x)
            return text[:-2] if text.endswith(".0") else text
    text = value.GetValue()
    if text is None:
        text = value.GetSummary() or "?"
    return text


def matrix_summary(valobj, internal_dict):
    layout = _Layout(valobj)
    if layout.error:
        return layout.error
    if not layout.valid:
        return "<no data>"
    if layout.size == 0:
        return "%dx%d []" % (layout.rows, layout.cols) if not layout.is_vector else "0 []"

    raw = valobj.GetNonSyntheticValue()

    def cell(i, j):
        return _format(layout.element(raw, "x", i, j))

    if layout.is_vector:
        n = layout.size
        shown = min(n, SUMMARY_MAX_COLS * SUMMARY_MAX_ROWS)
        cells = [cell(k, 0) if layout.cols == 1 else cell(0, k) for k in range(shown)]
        return "%d [%s%s]" % (n, ", ".join(cells), ", ..." if shown < n else "")

    rows = []
    for i in range(min(layout.rows, SUMMARY_MAX_ROWS)):
        cols = min(layout.cols, SUMMARY_MAX_COLS)
        cells = [cell(i, j) for j in range(cols)]
        rows.append("[%s%s]" % (", ".join(cells), ", ..." if cols < layout.cols else ""))
    more = ", ..." if layout.rows > SUMMARY_MAX_ROWS else ""
    return "%dx%d [%s%s]" % (layout.rows, layout.cols, ", ".join(rows), more)


class MatrixChildren:
    """Synthetic children: [i] for vectors, [i,j] in row order for matrices."""

    def __init__(self, valobj, internal_dict):
        self.valobj = valobj
        self.layout = None

    def update(self):
        self.layout = _Layout(self.valobj)
        return False

    def num_children(self):
        if not self.layout or not self.layout.valid or self.layout.error:
            return 0
        return min(self.layout.size, CHILDREN_MAX)

    def get_child_index(self, name):
        match = re.match(r"^\[(\d+)(?:,(\d+))?\]$", name)
        if not match:
            return -1
        if match.group(2) is None:
            return int(match.group(1))
        return int(match.group(1)) * self.layout.cols + int(match.group(2))

    def get_child_at_index(self, index):
        layout = self.layout
        if index < 0 or index >= self.num_children():
            return None
        raw = self.valobj.GetNonSyntheticValue()
        if layout.is_vector:
            i, j = (index, 0) if layout.cols == 1 else (0, index)
            return layout.element(raw, "[%d]" % index, i, j)
        i, j = divmod(index, layout.cols)
        return layout.element(raw, "[%d,%d]" % (i, j), i, j)

    def has_children(self):
        return True


_TYPES = [
    r"^Eigen::(Matrix|Array)<.+>$",
    r"^Eigen::Map<(const )?Eigen::(Matrix|Array)<.+>$",
]


def __lldb_init_module(debugger, internal_dict):
    for pattern in _TYPES:
        debugger.HandleCommand('type summary add -w eigen -F eigen_lldb.matrix_summary -x "%s"' % pattern)
        debugger.HandleCommand('type synthetic add -w eigen -l eigen_lldb.MatrixChildren -x "%s"' % pattern)
    debugger.HandleCommand("type category enable eigen")
