#    Research Resources
#    ------------------------------------
# -> uint vs int for indexing speeds: https://numba.discourse.group/t/uint64-vs-int64-indexing-performance-difference/1500
# -> binary vs linear search: https://dirtyhandscoding.wordpress.com/2017/08/25/performance-comparison-linear-search-vs-binary-search/
# -> fastmath for execution order insensitivity: https://numba.readthedocs.io/en/stable/user/performance-tips.html#fastmath
# -> other stuff already described in the mm_toolbox README :)

#    Improvements (Ongoing or to-do)
#    ------------------------------------
# -> faster normalization/denormalization: https://tbetcke.github.io/hpc_lecture_notes/working_with_numba.html (specifically for div)
# -> hybrid linear-binary search algorithms (cutoff point ~70?)
# -> bug-fixing for orderbook.display_internal() in denormalizing arrays prior to display
# -> get_bids() and get_asks() funcs which auto denormalize all arrays 
# -> perform orderbook indicators on pure uints before final, single denormalization
# -> change array signatures to specify C-style (change all int32[:, :] -> Array(int32, 2, "C"))
# -> add more here!
