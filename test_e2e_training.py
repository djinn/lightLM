import lightlm

try:
    args = lightlm.Args()
    args.input = "test_data.txt"
    args.minCount = 1
    args.minCountLabel = 1

    print("Creating LightLM object...")
    llm = lightlm.LightLM()

    print("Training...")
    llm.train(args)

    print("Training finished successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
