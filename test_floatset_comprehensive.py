#!/usr/bin/env python3
"""Comprehensive tests for FloatInterval and FloatSet classes."""

from geo.floatset import FloatInterval, FloatSet


def test_comprehensive():
    """Comprehensive test of all operations."""
    print("Comprehensive FloatInterval and FloatSet Tests")
    print("=" * 60)
    
    # Test 1: Basic interval operations
    print("\n1. Basic Interval Operations:")
    a = FloatInterval(0, 5)
    b = FloatInterval(2, 7)
    c = FloatInterval(8, 10)
    
    print(f"a = {a}, b = {b}, c = {c}")
    print(f"a ∩ b = {a & b}")
    print(f"a ∪ b = {a | b}")
    print(f"a - b = {a - b}")
    print(f"b - a = {b - a}")
    print(f"a Δ b = {a ^ b}")
    
    # Test 2: Disjoint intervals
    print(f"\na ∪ c (disjoint) = {a | c}")
    print(f"a ∩ c (disjoint) = {a & c}")
    print(f"a - c = {a - c}")
    print(f"c - a = {c - a}")
    
    # Test 3: Empty intervals
    print("\n2. Empty Interval Tests:")
    empty = FloatInterval(1, 0)
    print(f"empty = {empty}, bool(empty) = {bool(empty)}")
    print(f"a ∩ empty = {a & empty}")
    print(f"a ∪ empty = {a | empty}")
    print(f"empty ∪ a = {empty | a}")
    print(f"a - empty = {a - empty}")
    print(f"empty - a = {empty - a}")
    
    # Test 4: Single point intervals
    print("\n3. Single Point Intervals:")
    point = FloatInterval(3, 3)
    print(f"point = {point}, length = {point.length()}")
    print(f"3 in point = {3 in point}")
    print(f"3.1 in point = {3.1 in point}")
    print(f"a ∩ point = {a & point}")
    print(f"point ∩ a = {point & a}")
    
    # Test 5: FloatSet operations
    print("\n4. FloatSet Operations:")
    set1 = FloatSet([FloatInterval(0, 2), FloatInterval(4, 6)])
    set2 = FloatSet([FloatInterval(1, 5)])
    set3 = FloatSet([FloatInterval(7, 8)])
    
    print(f"set1 = {set1}")
    print(f"set2 = {set2}")
    print(f"set3 = {set3}")
    
    print(f"\nset1 ∪ set2 = {set1 | set2}")
    print(f"set1 ∩ set2 = {set1 & set2}")
    print(f"set1 - set2 = {set1 - set2}")
    print(f"set2 - set1 = {set2 - set1}")
    print(f"set1 Δ set2 = {set1 ^ set2}")
    
    # Test 6: Multiple intervals in sets
    print("\n5. Multiple Interval Sets:")
    complex_set = FloatSet([
        FloatInterval(0, 1),
        FloatInterval(1.5, 2.5),
        FloatInterval(3, 4),
        FloatInterval(4.5, 5.5)
    ])
    subtract_set = FloatSet([
        FloatInterval(0.5, 1.5),
        FloatInterval(2, 3.5),
        FloatInterval(5, 6)
    ])
    
    print(f"complex_set = {complex_set}")
    print(f"subtract_set = {subtract_set}")
    print(f"complex_set - subtract_set = {complex_set - subtract_set}")
    
    # Test 7: Edge cases with floating point precision
    print("\n6. Floating Point Precision:")
    eps = 1e-10
    tiny1 = FloatInterval(0, 1 - eps)
    tiny2 = FloatInterval(1 + eps, 2)
    print(f"tiny1 = {tiny1}, tiny2 = {tiny2}")
    print(f"tiny1 ∪ tiny2 = {tiny1 | tiny2}")
    print(f"tiny1 ∩ tiny2 = {tiny1 & tiny2}")
    
    # Test 8: Contains and subset tests
    print("\n7. Contains and Subset Tests:")
    big_interval = FloatInterval(0, 10)
    small_interval = FloatInterval(3, 7)
    print(f"{small_interval} ⊆ {big_interval} = {small_interval.is_subset(big_interval)}")
    
    big_set = FloatSet([FloatInterval(0, 5), FloatInterval(6, 10)])
    small_set = FloatSet([FloatInterval(1, 3), FloatInterval(7, 9)])
    print(f"{small_set} ⊆ {big_set} = {all(iv.is_subset(big_interval) for iv in small_set for big_interval in big_set)}")
    
    # Test 9: Normalization
    print("\n8. Normalization Tests:")
    unnormalized = [
        FloatInterval(5, 6),
        FloatInterval(1, 3),
        FloatInterval(2, 4),
        FloatInterval(0, 1)
    ]
    normalized_set = FloatSet(unnormalized)
    print(f"Unnormalized: {unnormalized}")
    print(f"Normalized: {normalized_set}")
    
    # Test 10: Empty set
    print("\n9. Empty Set Tests:")
    empty_set = FloatSet()
    print(f"empty_set = {empty_set}, bool(empty_set) = {bool(empty_set)}")
    print(f"set1 ∪ empty_set = {set1 | empty_set}")
    print(f"set1 ∩ empty_set = {set1 & empty_set}")
    print(f"empty_set ∪ set1 = {empty_set | set1}")
    print(f"empty_set ∩ set1 = {empty_set & set1}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_comprehensive()