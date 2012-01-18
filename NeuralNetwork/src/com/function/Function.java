/**
 * The function interface. All function objects should implement this interface.
 * Calculates the output from the implemented function.
 * 
 * @author Chao Zhang
 */
package com.function;

public interface Function {
    /**
     * Using the implemented function, calculates a value given an input.
     * 
     * @param input
     *            The input given to the implemented function for calculation.
     * @return The output returned from the implemented function.
     */
    public double calculate(double input);

    /**
     * Using the implemented function's mathematical derivative, calculates a
     * value given an input.
     * 
     * @param input
     *            The input to the implemented function's derivative for
     *            calculation.
     * @return The output returned from the implemented function's derivative.
     */
    public double calculateFirstDeriv(double input);
}
