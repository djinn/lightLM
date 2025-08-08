import lightlm

try:
    m = lightlm.Matrix(10, 20)
    print(f"Successfully created lightlm.Matrix of size ({m.rows}, {m.cols})")

    m.zero()
    print("Successfully called m.zero()")

except Exception as e:
    print(f"An error occurred: {e}")
