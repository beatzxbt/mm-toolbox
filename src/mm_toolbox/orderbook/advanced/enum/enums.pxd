cpdef enum CyOrderbookSortedness:
    UNKNOWN
    ASCENDING
    DESCENDING
    BIDS_ASCENDING_ASKS_DESCENDING
    BIDS_DESCENDING_ASKS_ASCENDING

cpdef CyOrderbookSortedness py_to_cy_orderbook_sortedness(object sortedness)


