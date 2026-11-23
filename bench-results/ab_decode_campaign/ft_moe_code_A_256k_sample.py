def mergesort(arr: list[int]) -> list[int]:
    """
    Sort a list of integers using the merge sort algorithm.

    Merge sort is a divide-and-conquer algorithm that divides the input list
    into two halves, recursively sorts each half, and then merges the two
    sorted halves to produce the sorted result.

    Args:
        arr: A list of integers to be sorted.

    Returns:
        A new list containing the sorted integers in ascending order.
    """
    if len(arr) <= 1:
        return arr

    # Divide the list into two halves
    mid = len(arr) // 2
    left_half = mergesort(arr[:mid])
    right_half = mergesort(arr[mid:])

    # Merge the sorted halves
    return _merge(left_half, right_half)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """
    Merge two sorted lists into a single sorted list.

    Args:
        left: A sorted list of integers.
        right: A sorted list of integers.

    Returns:
        A new list containing all elements from left and right in ascending order.
    """
    merged = []
    i = 0  # Pointer for left list
    j = 0  # Pointer for right list

    # Compare elements from both lists and add the smaller one to merged
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Append any remaining elements from left list
    while i < len(left):
        merged.append(left[i])
        i += 1

    # Append any remaining elements from right list
    while j < len(right):
        merged.append(right[j])
        j += 1

    return merged


if __name__ == "__main__":
    # Demo usage
    test_list = [38, 27, 43, 3, 9, 82, 10]
    print(f"Original list: {test_list}")
    sorted_list = mergesort(test_list)
    print(f"Sorted list:   {sorted_list}")

    # Additional test cases
    empty_list = []
    print(f"Empty list:    {mergesort(empty_list)}")

    single_element = [42]
    print(f"Single element: {mergesort(single_element)}")

    already_sorted = [1, 2, 3, 4, 5]
    print(f"Already sorted: {mergesort(already_sorted)}")

    reverse_sorted = [5, 4, 3, 2, 1]
    print(f"Reverse sorted: {mergesort(reverse_sorted)}")

    duplicates = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    print(f"With duplicates: {mergesort(duplicates)}")
