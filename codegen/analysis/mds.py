#!/usr/bin/env python3
# This script uses the galois library for finite field arithmetic
# Install: pip install galois numpy

import numpy as np
import galois
from itertools import combinations


# Helper function for manual determinant calculation (for matrices without .det() method)
def __manual_det(mat, field):
    """Calculate determinant manually for small matrices"""
    J = mat.shape[0]
    if J == 1:
        return field(mat[0, 0])
    elif J == 2:
        return field(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    else:
        det = field(0)
        for j in range(J):
            minor = np.delete(np.delete(mat, 0, axis=0), j, axis=1)
            minor_det = __manual_det(minor, field)
            cofactor = ((-1) ** j) * minor_det
            det += field(mat[0, j]) * cofactor
        return det


def check_mds(f, mat):
    """
    Check if a matrix is MDS over the field F_2[x]/<f>
    
    Args:
        f: tuple of exponents representing the defining polynomial (e.g., (0,1,3) for x^3 + x + 1)
        mat: J*L matrix where each entry is a tuple of exponents representing a polynomial in F_2[x]/<f>
             - Zero element is represented by an empty tuple ()
             - Non-zero elements are tuples of exponents (e.g., (0,) for 1, (1,) for x, (0,1) for 1+x)
    
    Returns:
        bool: True if the matrix is MDS, False otherwise
    """
    # Create the base field F_2
    F2 = galois.GF(2)
    
    # Convert tuple of exponents to a polynomial over F_2
    # Determine the degree of the polynomial
    max_degree = max(f) if f else 0
    coeffs = [1 if i in f else 0 for i in range(max_degree + 1)]
    polynomial_f = galois.Poly(coeffs, field=F2)
    
    # Check if f is irreducible
    if not polynomial_f.is_irreducible():
        raise ValueError(f"Defining polynomial {polynomial_f} is not irreducible over F_2")
    
    # Create the extension field F_2[x]/<f>
    degree = polynomial_f.degree
    if degree == 0:
        # Constant polynomial, which can't be irreducible unless it's 1, but 1 is not irreducible
        raise ValueError("Defining polynomial must be non-constant")
    
    # Create the appropriate field
    if degree == 1:
        # For degree 1, it's just the prime field GF(2)
        K = galois.GF(2)
    else:
        # For higher degrees, create the extension field GF(2^degree)
        # First try with the specified irreducible polynomial
        try:
            K = galois.GF(2**degree, irreducible_poly=polynomial_f)
        except ValueError:
            # If that fails, let galois choose the default irreducible polynomial for GF(2^degree)
            print(f"Using default irreducible polynomial for GF(2^{degree})")
            K = galois.GF(2**degree)
            print(f"Default polynomial: {K.irreducible_poly}")
    
    # Get matrix dimensions
    J = len(mat)
    if J == 0:
        raise ValueError("Matrix must have at least one row")
    
    L = len(mat[0])
    if L < J:
        raise ValueError("Matrix must have at least as many columns as rows (L >= J)")
    
    # Helper function to convert exponent tuple to field element
    def exponents_to_field_element(exponents):
        if not exponents:
            return K(0)
        
        # For GF(2), it's simple - sum the exponents mod 2
        if K.order == 2:
            return K(1) if len(exponents) % 2 == 1 else K(0)
        
        # For extension fields, build the field element directly
        # Each exponent represents a power of the primitive element
        result = K(0)
        for e in exponents:
            if e >= K.degree:
                # Reduce the exponent modulo the field polynomial's degree
                # Since we're in GF(2^m), x^m = p(x) where p is the irreducible polynomial
                # So we can reduce higher exponents using the field's multiplicative structure
                element = K.primitive_element**e
            else:
                # For exponents less than the degree, use the primitive element's powers directly
                element = K.primitive_element**e
            result += element
        
        return result
    
    # Convert the input matrix to a numpy array over field K
    galois_mat = np.empty((J, L), dtype=K)
    for i, row in enumerate(mat):
        for j, entry in enumerate(row):
            galois_mat[i, j] = exponents_to_field_element(entry)
    
    # Check all J*J submatrices
    for cols_indices in combinations(range(L), J):
        # Extract the submatrix
        submat = galois_mat[:, list(cols_indices)]
        
        # For 1x1 matrices, determinant is the only element
        if J == 1:
            det = submat[0, 0]
        elif J == 2:
            # For 2x2 matrices: det = ad - bc
            a = submat[0, 0]
            b = submat[0, 1]
            c = submat[1, 0]
            d = submat[1, 1]
            det = a * d - b * c
        elif J == 3:
            # For 3x3 matrices (Sarrus rule)
            a, b, c = submat[0, 0], submat[0, 1], submat[0, 2]
            d, e, f = submat[1, 0], submat[1, 1], submat[1, 2]
            g, h, i = submat[2, 0], submat[2, 1], submat[2, 2]
            det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
        else:
            # For larger matrices, use recursive determinant calculation
            det = __manual_det(submat, K)
        # Check if determinant is zero
        if det == K(0):
            return False
    # All J*J submatrices have nonzero determinant
    return True


# Example usage
if __name__ == "__main__":
    print("Checking MDS property using galois library...")
    print("=" * 50)
    
    try:
        # Example 1: Simple MDS matrix over F_2[x]/(x+1) = F_2
        # Define polynomial f = x + 1 (represented as (0,1))
        f1 = (0, 1)  # x + 1
        # 2x3 matrix over F_2
        mat1 = [
            [(0,), (1,), (0,)],  # [1, x, 0] but x=1 in F_2
            [(1,), (0,), (1,)]   # [x, 0, x] but x=1 in F_2
        ]
        print("Example 1 - Simple 2x3 binary matrix:")
        print(f"Defining polynomial: x + 1")
        print(f"Matrix:\n{np.array(mat1, dtype=object)}")
        result1 = check_mds(f1, mat1)
        print(f"Is MDS: {result1}")
        print()
        
        # Example 2: Matrix over F_4 = F_2[x]/(x^2 + x + 1)
        # Define polynomial f = x^2 + x + 1 (represented as (0,1,2))
        f2 = (0, 1, 2)  # x^2 + x + 1
        # 2x3 matrix over F_4
        mat2 = [
            [(0,), (1,), (2,)],      # [1, x, x^2]
            [(1,), (2,), (0, 1)]     # [x, x^2, 1 + x]
        ]
        print("Example 2 - 2x3 matrix over F_4:")
        print(f"Defining polynomial: x^2 + x + 1")
        print(f"Matrix:\n{np.array(mat2, dtype=object)}")
        result2 = check_mds(f2, mat2)
        print(f"Is MDS: {result2}")
        print()
        
        # Example 3: Non-MDS matrix
        print("Example 3 - Non-MDS matrix (same polynomial as Example 2):")
        non_mds_mat = [
            [(0,), (1,), (0,)],      # [1, x, 0]
            [(1,), (0,), (1,)]       # [x, 0, x]
        ]
        print(f"Matrix:\n{np.array(non_mds_mat, dtype=object)}")
        result3 = check_mds(f2, non_mds_mat)
        print(f"Is MDS: {result3}")
        print()
        
        # Example 4: Known MDS matrix (2x2 identity matrix over F_2)
        print("Example 4 - Known MDS matrix (2x2 identity matrix over F_2):")
        f4 = (0, 1)  # x + 1 = F_2
        mds_mat = [
            [(0,), ()],  # [1, 0] in F_2
            [(), (0,)]   # [0, 1] in F_2
        ]
        print(f"Defining polynomial: x + 1")
        print(f"Matrix:\n{np.array(mds_mat, dtype=object)}")
        result4 = check_mds(f4, mds_mat)
        print(f"Is MDS: {result4}")
        
        # Example 5: Simple 2x2 matrix with non-zero determinant
        print("\nExample 5 - Simple 2x2 matrix with non-zero determinant over F_2:")
        f5 = (0, 1)  # x + 1 = F_2
        simple_mat = [
            [(0,), (0,)],  # [1, 1] in F_2
            [(0,), ()]     # [1, 0] in F_2
        ]
        print(f"Defining polynomial: x + 1")
        print(f"Matrix:\n{np.array(simple_mat, dtype=object)}")
        result5 = check_mds(f5, simple_mat)
        print(f"Is MDS: {result5}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
