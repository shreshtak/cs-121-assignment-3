# TODO
#   - Build partial indexes and merge them. From lec 17:
#       - Build the inverted list structure until a certain size
#           – Which size?
#       - Then write the partial index to disk, start making a new one
#       - At the end of this process, the disk is filled with many partial
#           indexes, which are merged
#       - Partial lists (=partial indexes) must be designed so they can be
#           merged in small pieces
#           – e.g., storing in alphabetical order
#   - Using the above index, process AND queries (at minimum)
#       - See lecture 17 slide 25
