import milestone1 as m1

def mergePartialIndexes():
    parsed = {}
    with open('inverted_index.txt', 'r') as f: 
        for line in f: # each line is a partial index
            line = line.split(":")
            parsed[line[0]] = line[1] # parse the indexes

            # implement priority queue
            

#       - Partial lists (=partial indexes) must be designed so they can be
#           merged in small pieces
#           – e.g., storing in alphabetical order
#   - Using the above index, process AND queries (at minimum)
#       - See lecture 17 slide 25


# 1. Read the Partial Indexes: Open the text file and read each partial index. 
#    If they are separated by new lines, you can read the file line by line.

# 2. Parse the Indexes: For each line (which represents a partial index), parse the token and its posting list. 
#    You'll need to decide on a format for how tokens and posting lists are represented in each line.

# 3. Initialize a Priority Queue: Create a priority queue to keep track of the smallest token from 
#    each partial index's current position.

# 4. Merge: Perform the merge by repeatedly extracting the minimum token from the priority queue, 
#    merging posting lists when necessary (if the same token appears in multiple partial indexes), 
#    and writing the merged token and its posting list to the output file.

# 5. Write the Merged Index: As you extract and merge tokens and their posting lists, write them to a new file that will become your merged index.
