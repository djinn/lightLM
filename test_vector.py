import lightlm

try:
    v = lightlm.Vector(10)
    print(f"Successfully created lightlm.Vector of size {len(v)}")

    # At this point, the vector is uninitialized.
    # Let's call the zero() method to initialize it.
    v.zero()
    print("Successfully called v.zero()")

    # I can't access the elements directly yet, as I haven't implemented
    # the buffer protocol or `__getitem__`. But I can verify the methods work.

except Exception as e:
    print(f"An error occurred: {e}")
