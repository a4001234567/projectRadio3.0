def poly_divmod(a, b):
     """
     Perform polynomial division in GF(2).
     Args:
         a (int): Dividend polynomial as an integer.
         b (int): Divisor polynomial as an integer.
 
     Returns:
         tuple: (quotient, remainder), both as integers.
     """
     if b == 0:
         raise ValueError("Division by zero polynomial is undefined.")
     
     a_deg = a.bit_length() - 1  # Degree of a
     b_deg = b.bit_length() - 1  # Degree of b
     
     quotient = 0
     remainder = a
 
     while remainder != 0 and (remainder.bit_length() - 1) >= b_deg:
         shift = (remainder.bit_length() - 1) - b_deg
         quotient ^= (1 << shift)  # Add this term to the quotient
         remainder ^= (b << shift)  # Subtract b(x) shifted by `shift`
     
     return quotient, remainder
 
 
def poly_gcd(a, b):
     """
     Compute the GCD of two polynomials in GF(2).
     Polynomials are represented as integers, where the binary representation corresponds to coefficients.
 
     Args:
         a (int): Polynomial A as an integer.
         b (int): Polynomial B as an integer.
 
     Returns:
         int: GCD of the polynomials as an integer.
     """
     while b != 0:
         _, remainder = poly_divmod(a, b)
         a, b = b, remainder
     return a
 
 
def polynomial_repr(p):
     """
     Print the polynomial in human-readable form.
 
     Args:
         p (int): Polynomial as an integer.
     """
     if p == 0:
         return "0"

     terms = []
     i = 0
     while p:
         if p & 1:
             if i == 0:
                 terms.append("1")
             elif i == 1:
                 terms.append("x")
             else:
                 terms.append(f"x^{i}")
         p >>= 1
         i += 1
     
     return " + ".join(reversed(terms))

def poly_to_binary(poly_str):
     """
     Convert a polynomial string representation into its binary number representation.
 
     Args:
         poly_str (str): Polynomial in string form, e.g., "x^3 + x + 1".
 
     Returns:
         int: Binary representation of the polynomial as an integer.
     """
     poly_str = poly_str.replace(" ", "").split("+")  # Remove spaces and split terms
     binary = 0
 
     for term in poly_str:
         if term == "1":  # Constant term
             binary |= 1
         elif term == "x":  # x^1 term
             binary |= (1 << 1)
         elif term.startswith("x^"):  # Higher degree terms
             degree = int(term[2:])  # Extract the degree
             binary |= (1 << degree)
     
     return binary
 
# Example Usage
if __name__ == "__main__":
     # Example: Polynomials x^3 + x + 1 (binary: 1011) and x^2 + x (binary: 110)
     poly1 = 0b101101
     poly2 = 0b101
     poly1 = (1<<127)+1
     poly2 = 0b10010001 
 
     gcd = poly_gcd(poly1, poly2)
     print("GCD of the polynomials:")
     print_polynomial(gcd)

