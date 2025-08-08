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

    # 2. Create quantization args
    qargs = lightlm.Args()
    qargs.dsub = 2
    qargs.qnorm = True

    # 3. Quantize the model
    print("Quantizing...")
    llm.quantize(qargs)
    print("Quantization finished successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
