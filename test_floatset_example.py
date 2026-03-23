#!/usr/bin/env python3
"""Example usage of FloatInterval and FloatSet classes."""

from geo.floatset import FloatInterval, FloatSet


def test_float_interval():
    """Test FloatInterval class."""
    print("Testing FloatInterval:")
    print("-" * 40)
    
    # Create intervals
    iv1 = FloatInterval(0.0, 1.0)
    iv2 = FloatInterval(0.5, 2.0)
    iv3 = FloatInterval(3.0, 4.0)
    empty = FloatInterval(1.0, 0.0)  # Empty interval
    
    print(f"iv1 = {iv1}")
    print(f"iv2 = {iv2}")
    print(f"iv3 = {iv3}")
    print(f"empty = {empty}")
    print(f"empty.is_empty() = {empty.is_empty()}")
    print(f"iv1.is_empty() = {iv1.is_empty()}")
    
    # Test contains
    print(f"\n0.7 in iv1 = {0.7 in iv1}")
    print(f"1.5 in iv1 = {1.5 in iv1}")
    
    # Test intersection
    inter = iv1.intersection(iv2)
    print(f"\niv1 ∩ iv2 = {inter}")
    
    # Test union
    union = iv1.union(iv2)
    print(f"iv1 ∪ iv2 = {union}")
    
    # Test difference
    diff = iv1.difference(iv2)
    print(f"iv1 - iv2 = {diff}")
    
    # Test symmetric difference
    sym_diff = iv1.symmetric_difference(iv2)
    print(f"iv1 Δ iv2 = {sym_diff}")
    
    # Test disjoint intervals
    union_disjoint = iv1.union(iv3)
    print(f"\niv1 ∪ iv3 (disjoint) = {union_disjoint}")
    
    # Test operators
    print(f"\nOperators:")
    print(f"iv1 & iv2 = {iv1 & iv2}")
    print(f"iv1 | iv2 = {iv1 | iv2}")
    print(f"iv1 - iv2 = {iv1 - iv2}")
    print(f"iv1 ^ iv2 = {iv1 ^ iv2}")


def test_float_set():
    """Test FloatSet class."""
    print("\n\nTesting FloatSet:")
    print("-" * 40)
    
    # Create sets
    set1 = FloatSet([FloatInterval(0.0, 1.0), FloatInterval(2.0, 3.0)])
    set2 = FloatSet([FloatInterval(0.5, 2.5)])
    set3 = FloatSet([FloatInterval(4.0, 5.0)])
    
    print(f"set1 = {set1}")
    print(f"set2 = {set2}")
    print(f"set3 = {set3}")
    
    # Test union
    union = set1.union(set2)
    print(f"\nset1 ∪ set2 = {union}")
    
    # Test intersection
    inter = set1.intersection(set2)
    print(f"set1 ∩ set2 = {inter}")
    
    # Test difference
    diff = set1.difference(set2)
    print(f"set1 - set2 = {diff}")
    
    # Test symmetric difference
    sym_diff = set1.symmetric_difference(set2)
    print(f"set1 Δ set2 = {sym_diff}")
    
    # Test disjoint sets
    union_disjoint = set1.union(set3)
    print(f"\nset1 ∪ set3 (disjoint) = {union_disjoint}")
    
    # Test operators
    print(f"\nOperators:")
    print(f"set1 | set2 = {set1 | set2}")
    print(f"set1 & set2 = {set1 & set2}")
    print(f"set1 - set2 = {set1 - set2}")
    print(f"set1 ^ set2 = {set1 ^ set2}")
    
    # Test contains
    print(f"\n0.7 in set1 = {0.7 in set1}")
    print(f"1.5 in set1 = {1.5 in set1}")
    print(f"2.5 in set1 = {2.5 in set1}")
    
    # Test empty set
    empty_set = FloatSet()
    print(f"\nempty_set = {empty_set}")
    print(f"empty_set.is_empty() = {empty_set.is_empty()}")


def test_edge_cases():
    """Test edge cases."""
    print("\n\nTesting Edge Cases:")
    print("-" * 40)
    
    # Test touching intervals
    iv1 = FloatInterval(0.0, 1.0)
    iv2 = FloatInterval(1.0, 2.0)
    print(f"iv1 = {iv1}, iv2 = {iv2}")
    print(f"iv1 ∪ iv2 (touching) = {iv1.union(iv2)}")
    
    # Test almost touching intervals
    iv3 = FloatInterval(0.0, 0.999999)
    iv4 = FloatInterval(1.000001, 2.0)
    print(f"\niv3 = {iv3}, iv4 = {iv4}")
    print(f"iv3 ∪ iv4 (almost touching) = {iv3.union(iv4)}")
    
    # Test normalization
    intervals = [
        FloatInterval(2.0, 3.0),
        FloatInterval(1.0, 2.5),
        FloatInterval(0.0, 1.0)
    ]
    fset = FloatSet(intervals)
    print(f"\nUnnormalized intervals: {intervals}")
    print(f"Normalized set: {fset}")
    print(f"Intervals in set: {list(fset)}")


if __name__ == "__main__":
    test_float_interval()
    test_float_set()
    test_edge_cases()