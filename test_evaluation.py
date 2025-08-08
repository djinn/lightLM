import lightlm

try:
    # 1. Train a model
    args = lightlm.Args()
    args.input = "test_data.txt"
    args.minCount = 1
    args.minCountLabel = 1

    llm = lightlm.LightLM()
    llm.train(args)
    print("Training finished successfully.")

    # 2. Test the model
    print("Testing...")
    llm.test("test_data.txt")
    print("Testing finished successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
