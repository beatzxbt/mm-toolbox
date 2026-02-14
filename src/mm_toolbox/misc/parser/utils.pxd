# Parse an unsigned integer from ASCII digits until `delimiter`.
# Contract:
# - `cursor[0]` must point at the first digit.
# - On return, `cursor[0]` points at `delimiter`.
cdef inline unsigned long long parse_u64_until_char(
    const unsigned char** cursor,
    unsigned char delimiter,
) noexcept:
    cdef:
        unsigned long long value = 0
        const unsigned char* ptr = cursor[0]

    while ptr[0] != delimiter:
        value = (value * 10) + <unsigned long long>(ptr[0] - 48)
        ptr += 1

    cursor[0] = ptr
    return value


# Parse a quoted ASCII span without allocations.
# Contract:
# - `cursor[0]` must point at the first character inside quotes.
# - `start[0]`/`end[0]` are set to the span [start, end).
# - On return, `cursor[0]` points at the closing quote (`"`).
cdef inline void parse_quoted_span(
    const unsigned char** cursor,
    const char** start,
    const char** end,
) noexcept:
    cdef const unsigned char* ptr = cursor[0]
    start[0] = <const char*>ptr
    while ptr[0] != 34:  # '"'
        ptr += 1
    end[0] = <const char*>ptr
    cursor[0] = ptr


# Fast check for the exact token `MARKET"` at `cursor`.
# Used as a hot-path branch in the trade parser before generic intern logic.
cdef inline bint is_market_token(const unsigned char* cursor) noexcept:
    return (
        cursor[0] == 77  # 'M'
        and cursor[1] == 65  # 'A'
        and cursor[2] == 82  # 'R'
        and cursor[3] == 75  # 'K'
        and cursor[4] == 69  # 'E'
        and cursor[5] == 84  # 'T'
        and cursor[6] == 34  # '"'
    )
