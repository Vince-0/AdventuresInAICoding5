def mergesort(arr: list[int]) -> list[int]:
    """
    Sort a list of integers using the mergesort algorithm.

    Mergesort is a divide-and-conquer algorithm that works by recursively
    splitting the input list into halves, sorting each half, and then merging
    the two sorted halves back together. It has a time complexity of O(n log n)
    in all cases (best, average, and worst) and a space complexity of O(n).

    Args:
        arr: A list of integers to be sorted.

    Returns:
        A new list containing the sorted integers in ascending order.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = mergesort(arr[:mid])
    right_half = mergesort(arr[mid:])

    return _merge(left_half, right_half)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """
    Merge two sorted lists into a single sorted list.

    Args:
        left: A sorted list of integers.
        right: A sorted list of integers.

    Returns:
        A new list containing all elements from left and right in sorted order.
    """
    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append any remaining elements from left
    while i < len(left):
        result.append(left[i])
        i += 1

    # Append any remaining elements from right
    while j < len(right):
        result.append(right[j])
        j += 1

    return result


if __name__ == "__main__":
    # Demo with various test cases
    test_cases = [
        [5, 2, 8, 1, 9, 3, 7, 4, 6],
        [1],
        [],
        [3, 3, 3, 3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5],
        [-5, 10, -3, 0, 7, -1, 4],
    ]

    for test in test_cases:
        original = test.copy()
        sorted_list = mergesort(test)
        print(f"Original: {original}")
        print(f"Sorted:   {sorted_list}")
        print()
