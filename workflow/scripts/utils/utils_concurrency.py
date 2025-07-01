
######################################################
# BATCH CREATION
######################################################
def split_list(values, size):
    """
    Split given list into chunks of given size
    """
    values = list(values)
    for i in range(0, len(values), size):
        yield values[i:(i + size)]

def split_by_size(input, n):
    """
    Split the input in different groups of size N
    :param input: (int) length of the object
    :param n : (int) size of each group

    :return list of tuples with the start and the end of each group
    """

    ranges = [(i * n, (i + 1) * n)  for i in range((input + n - 1) // n ) ]
    # Adjust the last range to the length of the object
    ranges[-1] = (ranges[-1][0], input)

    return ranges