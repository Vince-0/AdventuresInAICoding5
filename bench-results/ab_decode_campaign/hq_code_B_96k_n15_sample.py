def mergesort(arr: list[int]) -> list[int]:
    """
    Sort a list of integers using the mergesort algorithm.
    
    Mergesort is a divide-and-conquer algorithm that works by recursively
    splitting the input list into smaller sublists until each sublist contains
    a single element, then merging those sublists back together in sorted order.
    
    Time Complexity: O(n log n) in all cases (best, average, worst).
    Space Complexity: O(n) due to the temporary lists created during merging.
    
    Args:
        arr: A list of integers to be sorted.
        
    Returns:
        A new list containing the sorted integers.
        
    Examples:
        >>> mergesort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
        >>> mergesort([])
        []
        >>> mergesort([42])
        [42]
        >>> mergesort([5, 3, 8, 1, 9, 2, 7, 4, 6])
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
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
    merged = []
    i = 0
    j = 0
    
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
    # Demo with various test cases
    test_cases = [
        [3, 1, 4, 1, 5, 9, 2, 6],
        [],
        [42],
        [5, 3, 8, 1, 9, 2, 7, 4, 6],
        [10, -1, 0, 5, -3, 7, 2, -8, 4],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [99, 88, 77, 66, 55, 44, 33, 22, 11, 0],
    ]
    
    print("Mergesort Demo")
    print("=" * 50)
    
    for test in test_cases:
        original = test.copy()
        sorted_result = mergesort(test)
        print(f"Input:  {original}")
        print(f"Sorted: {sorted_result}")
        print("-" * 50)
    
    # Verify correctness against Python's built-in sort
    print("\nVerification against built-in sort:")
    print("=" * 50)
    
    import random
    
    for _ in range(5):
        random_list = [random.randint(-100, 100) for _ in range(20)]
        expected = sorted(random_list)
        result = mergesort(random_list)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: {random_list} -> {result}")
    
    print("\nAll tests completed.")
