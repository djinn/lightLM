import lightlm

try:
    args = lightlm.Args()
    print("Successfully created lightlm.Args() object.")
    # We can't access the attributes yet, because we haven't defined
    # the getters and setters in the C extension.
    # But we can verify that the object was created.
    print(args)
except Exception as e:
    print(f"An error occurred: {e}")
