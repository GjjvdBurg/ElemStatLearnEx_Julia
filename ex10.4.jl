
using DecisionTree

# copied from DecisionTree/src/measures.jl
function _weighted_error{T<:Real}(actual::Vector, predicted::Vector,
    weights::Vector{T})
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end

function make_adaboost_plotdata(trainX::Matrix, trainY::Vector, testX::Matrix,
    testY::Vector, total_iterations=100)
    # initialize
    N_train = length(trainY)
    N_test = length(testY)
    weights = ones(N_train)/N_train
    stumps = Node[]
    coeffs = FloatingPoint[]
    test_errors = FloatingPoint[]
    train_errors = FloatingPoint[]

    # run AdaBoost iterations
    for i in 1:total_iterations
        new_stump = build_stump(trainY, trainX, weights)
        predictions = apply_tree(new_stump, trainX)
        err = _weighted_error(trainY, predictions, weights)
        new_coeff = log((1 - err)/err)

        matches = trainY .== predictions
        weights[!matches] *= exp(new_coeff)
        weights /= sum(weights)

        push!(coeffs, new_coeff)
        push!(stumps, new_stump)

        train_predictions = apply_adaboost_stumps(Ensemble(stumps), coeffs,
                            trainX)
        test_predictions = apply_adaboost_stumps(Ensemble(stumps), coeffs,
                            testX)

        train_err = sum(train_predictions .!= trainY)/N_train
        test_err = sum(test_predictions .!= testY)/N_test
        @printf(".")

        push!(test_errors, test_err)
        push!(train_errors, train_err)
    end
    @printf("\n");
    return (test_errors, train_errors)
end

