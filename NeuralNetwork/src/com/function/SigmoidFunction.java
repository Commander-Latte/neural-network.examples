/**
 * An object representing the sigmoid function. Implemented with the function
 * interface.
 * 
 * @author Chao Zhang
 */
package com.function;

public class SigmoidFunction implements Function {
    /**
     * Calculates the output from the sigmoid function:
     * 
     * output = 1 / (1 + e^(-input))
     * 
     * @param input
     *            The input to the sigmoid function.
     * @return The output from the sigmoid function.
     */
    @Override
    public double calculate(double input) {
        return 1.0 / (1.0 + Math.pow(Math.E, -input));
    }

    /**
     * Calculates the output from the sigmoid function's first derivative:
     * 
     * output = input * (1 - input)
     * 
     * @param input
     *            The input to the sigmoid function's first derivative.
     * @return The output from the sigmoid function's first derivative.
     */
    @Override
    public double calculateFirstDeriv(double input) {
        return input * (1.0 - input);
    }
}
