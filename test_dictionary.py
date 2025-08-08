import lightlm

try:
    args = lightlm.Args()
    args.hello()

    d = lightlm.Dictionary(args)
    print("Successfully created lightlm.Dictionary object.")

    d.read_from_file("test_data.txt")
    print("Successfully called read_from_file().")

    print(f"Number of words: {d.nwords}")
    print(f"Number of labels: {d.nlabels}")
    print(f"Number of tokens: {d.ntokens}")

except Exception as e:
    print(f"An error occurred: {e}")
