def mergesort(arr: list[int]) -> list[int]:
    """
    Sort a list of integers using the mergesort algorithm.

    Mergesort is a divide-and-conquer algorithm that works by recursively
    splitting the input list into two halves, sorting each half, and then
    merging the two sorted halves into a single sorted list.

    Time Complexity: O(n log n) in all cases (best, average, worst).
    Space Complexity: O(n) due to the temporary lists created during merging.

    Args:
        arr: A list of integers to be sorted.

    Returns:
        A new list containing the sorted integers in ascending order.
        The original list is not modified.

    Examples:
        >>> mergesort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
        >>> mergesort([])
        []
        >>> mergesort([42])
        [42]
        >>> mergesort([5, 3, 8, 1, 2])
        [1, 2, 3, 5, 8]
    """
    # Base case: a list of 0 or 1 elements is already sorted
    if len(arr) <= 1:
        return arr[:]

    # Divide: split the list into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Conquer: recursively sort each half
    left_sorted = mergesort(left_half)
    right_sorted = mergesort(right_half)

    # Combine: merge the two sorted halves
    return _merge(left_sorted, right_sorted)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """
    Merge two sorted lists into a single sorted list.

    Args:
        left: A sorted list of integers.
        right: A sorted list of integers.

    Returns:
        A new sorted list containing all elements from left and right.
    """
    merged = []
    i = 0  # Pointer for left list
    j = 0  # Pointer for right list

    # Compare elements from both lists and append the smaller one
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Append any remaining elements from left
    while i < len(left):
        merged.append(left[i])
        i += 1

    # Append any remaining elements from right
    while j < len(right):
        merged.append(right[j])
        j += 1

    return merged


if __name__ == "__main__":
    # Demo: test mergesort with various inputs
    test_cases = [
        ([3, 1, 4, 1, 5, 9, 2, 6], [1, 1, 2, 3, 4, 5, 6, 9]),
        ([], []),
        ([42], [42]),
        ([5, 3, 8, 1, 2], [1, 2, 3, 5, 8]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([-3, -1, -4, -1, -5], [-5, -4, -3, -1, -1]),
        ([0, 0, 0], [0, 0, 0]),
        ([10, -10, 20, -20, 30, -30], [-30, -20, -10, 10, 20, 30]),
    ]

    print("Mergesort Demo")
    print("=" * 50)

    all_passed = True
    for i, (input_list, expected) in enumerate(test_cases, 1):
        result = mergesort(input_list)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"Test {i}: {status}")
        print(f"  Input:    {input_list}")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print()

    print("=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
