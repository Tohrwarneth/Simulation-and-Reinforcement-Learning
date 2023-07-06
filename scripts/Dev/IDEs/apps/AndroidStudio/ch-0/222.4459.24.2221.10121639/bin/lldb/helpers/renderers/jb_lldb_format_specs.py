import lldb
from jb_declarative_formatters import TypeVizFormatSpec, TypeVizFormatFlags
from renderers.jb_lldb_utils import get_root_value, get_value_format

# @formatter:off
eFormatHexNoPrefix              = lldb.kNumFormats + 1
eFormatHexUppercaseNoPrefix     = lldb.kNumFormats + 2
eFormatBinaryNoPrefix           = lldb.kNumFormats + 3
eFormatCStringNoQuotes          = lldb.kNumFormats + 4
eFormatUtf8String               = lldb.kNumFormats + 5
eFormatUtf8StringNoQuotes       = lldb.kNumFormats + 6
eFormatWideString               = lldb.kNumFormats + 7
eFormatWideStringNoQuotes       = lldb.kNumFormats + 8
eFormatUtf32String              = lldb.kNumFormats + 9
eFormatUtf32StringNoQuotes      = lldb.kNumFormats + 10

eFormatBasicSpecsMask = (1 << 6) - 1
eFormatFlagSpecsMask = (1 << 20) - 1 - ((1 << 6) - 1)

eFormatNoAddress = 1 << 6
eFormatNoDerived = 1 << 7
eFormatNoRawView = 1 << 8
eFormatRawView   = 1 << 9
eFormatAsArray   = 1 << 10

eFormatInheritedFlagsMask = ~eFormatRawView
# @formatter:on

TYPE_VIZ_FORMAT_SPEC_TO_LLDB_FORMAT_MAP = {
    TypeVizFormatSpec.DECIMAL: lldb.eFormatDecimal,
    TypeVizFormatSpec.OCTAL: lldb.eFormatOctal,
    TypeVizFormatSpec.HEX: lldb.eFormatHex,
    TypeVizFormatSpec.HEX_UPPERCASE: lldb.eFormatHexUppercase,
    TypeVizFormatSpec.HEX_NO_PREFIX: eFormatHexNoPrefix,
    TypeVizFormatSpec.HEX_UPPERCASE_NO_PREFIX: eFormatHexUppercaseNoPrefix,
    TypeVizFormatSpec.BINARY: lldb.eFormatBinary,
    TypeVizFormatSpec.BINARY_NO_PREFIX: eFormatBinaryNoPrefix,
    TypeVizFormatSpec.SCIENTIFIC: lldb.eFormatFloat,  # TODO
    TypeVizFormatSpec.SCIENTIFIC_MIN: lldb.eFormatFloat,  # TODO
    TypeVizFormatSpec.CHARACTER: lldb.eFormatChar,
    TypeVizFormatSpec.STRING: lldb.eFormatCString,
    TypeVizFormatSpec.STRING_NO_QUOTES: eFormatCStringNoQuotes,
    TypeVizFormatSpec.UTF8_STRING: eFormatUtf8String,
    TypeVizFormatSpec.UTF8_STRING_NO_QUOTES: eFormatUtf8StringNoQuotes,
    TypeVizFormatSpec.WIDE_STRING: eFormatWideString,
    TypeVizFormatSpec.WIDE_STRING_NO_QUOTES: eFormatWideStringNoQuotes,
    TypeVizFormatSpec.UTF32_STRING: eFormatUtf32String,
    TypeVizFormatSpec.UTF32_STRING_NO_QUOTES: eFormatUtf32StringNoQuotes,
    TypeVizFormatSpec.ENUM: lldb.eFormatEnum,
    TypeVizFormatSpec.HEAP_ARRAY: lldb.eFormatDefault,  # TODO
    TypeVizFormatSpec.IGNORED: lldb.eFormatDefault,
}

TYPE_VIZ_FORMAT_FLAGS_TO_LLDB_FORMAT_MAP = {
    TypeVizFormatFlags.NO_ADDRESS: eFormatNoAddress,
    TypeVizFormatFlags.NO_DERIVED: eFormatNoDerived,
    TypeVizFormatFlags.NO_RAW_VIEW: eFormatNoRawView,
    TypeVizFormatFlags.NUMERIC_RAW_VIEW: lldb.eFormatDefault,  # TODO
    TypeVizFormatFlags.RAW_FORMAT: eFormatRawView,
}

FMT_STRING_SET = {lldb.eFormatCString: (1, "", '__locale__'),
                  eFormatUtf8String: (1, "", 'utf-8'),
                  eFormatWideString: (2, "L", 'utf-16'),
                  eFormatUtf32String: (4, "U", 'utf-32')}
FMT_STRING_NOQUOTES_SET = {eFormatCStringNoQuotes: (1, "", '__locale__'),
                           eFormatUtf8StringNoQuotes: (1, "", 'utf-8'),
                           eFormatWideStringNoQuotes: (2, "L", 'utf-16'),
                           eFormatUtf32StringNoQuotes: (4, "U", 'utf-32')}
FMT_STRING_SET_ALL = {**FMT_STRING_SET, **FMT_STRING_NOQUOTES_SET}

FMT_UNQUOTE_MAP = {
    lldb.eFormatCString: eFormatCStringNoQuotes,
    eFormatUtf8String: eFormatUtf8StringNoQuotes,
    eFormatWideString: eFormatWideStringNoQuotes,
    eFormatUtf32String: eFormatUtf32StringNoQuotes,
}


def get_custom_view_id(format_spec: int) -> int:
    return format_spec >> 20


def set_custom_view_id(format_spec: int, custom_view_spec=0) -> int:
    return format_spec | (custom_view_spec << 20)


def overlay_child_format(child: lldb.SBValue, parent_spec: int):
    child_root = get_root_value(child)
    child_spec = child_root.GetFormat()

    basic_specs = child_spec & eFormatBasicSpecsMask
    parent_basic_specs = parent_spec & eFormatBasicSpecsMask
    # TODO: more complex logic to merge basic specs
    if basic_specs == 0:
        basic_specs = parent_basic_specs

    flag_specs = (child_spec & eFormatFlagSpecsMask) | \
                 (parent_spec & eFormatFlagSpecsMask & eFormatInheritedFlagsMask)

    custom_view_spec = get_custom_view_id(child_spec)

    fmt = set_custom_view_id(basic_specs | flag_specs, custom_view_spec)
    child_root.SetFormat(fmt)


def overlay_summary_format(child: lldb.SBValue, parent_non_synth: lldb.SBValue):
    child_root = get_root_value(child)
    child_spec = child_root.GetFormat()
    parent_spec = parent_non_synth.GetFormat()

    basic_specs = child_spec & eFormatBasicSpecsMask
    parent_basic_specs = parent_spec & eFormatBasicSpecsMask

    if basic_specs == 0:
        basic_specs = parent_basic_specs
    elif basic_specs in FMT_UNQUOTE_MAP and parent_basic_specs in FMT_STRING_NOQUOTES_SET:
        # special case for FName
        basic_specs = FMT_UNQUOTE_MAP[basic_specs]

    flag_specs = (child_spec & eFormatFlagSpecsMask) | \
                 (parent_spec & eFormatFlagSpecsMask & eFormatInheritedFlagsMask)

    custom_view_spec = get_custom_view_id(child_spec)

    fmt = set_custom_view_id(basic_specs | flag_specs, custom_view_spec)
    if parent_spec & eFormatAsArray != 0 and child_spec & eFormatAsArray == 0:
        fmt |= eFormatAsArray
        size = parent_non_synth.GetFormatAsArraySize()
        child_root.SetFormatAsArraySize(size)

    child_root.SetFormat(fmt)


def update_value_dynamic_state(value: lldb.SBValue):
    fmt = get_value_format(value)
    if fmt & eFormatNoDerived:
        value.SetPreferDynamicValue(lldb.eNoDynamicValues)
