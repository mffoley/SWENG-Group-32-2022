simple model = 97%

simpleweighted = 85%

LSTM = 83.5%





Training Simple Model:

    train, test = readECGData()

    train = resampleData(train)

    train_inputs = formatInput(train) 
    test_inputs = formatInput(test) 
 
    train_inputs, test_inputs = reshapeInputs(train_inputs, test_inputs)
    train_outputs, test_outputs = formatOutputs(train, test)

    print(train_inputs.shape)
    print(test_inputs.shape)

    model = makeModel(train_inputs)

    trainModel(model, train_inputs, train_outputs, test_inputs, test_outputs)


Training LSTM model:


Training simple weighted:


    train, test = readECGData()

    raw_train_outputs = train[187]

    train_inputs = formatInput(train) 
    test_inputs = formatInput(test) 




    train_inputs= addGaussianNoise(train_inputs)

    train_inputs, test_inputs = reshapeInputs(train_inputs, test_inputs)
    train_outputs, test_outputs = formatOutputs(train, test)

    print(train_inputs.shape)
    print(train_inputs.shape)



    model = makeModel(train_inputs)

    trainModelClassWeight(model, train_inputs, train_outputs, test_inputs, test_outputs, raw_train_outputs)





Testing model:
    train, test = readECGData()

    train_inputs = formatInput(train) 
    test_inputs = formatInput(test) 




    # train_inputs= addGaussianNoise(train_inputs)

    train_inputs, test_inputs = reshapeInputs(train_inputs, test_inputs)
    train_outputs, test_outputs = formatOutputs(train, test)

    print(train_inputs.shape)
    print(train_inputs.shape)

    model = keras.models.load_model("simplemodel")

    model.evaluate(test_inputs, test_outputs)