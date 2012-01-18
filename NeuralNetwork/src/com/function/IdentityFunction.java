/**
 * The identity function. Always returns the input value.
 * 
 * @author Chao Zhang
 */

package com.function;

public class IdentityFunction implements Function {

    /**
     * Returns the input.
     * 
     * @param input
     *            The input to be returned.
     * @return The input.
     */
    @Override
    public double calculate(double input) {
        return input;
    }

    /**
     * Returns the mathematical derivative of the identity function; will always
     * return 1.
     * 
     * @param input
     *            The input to never be returned.
     * @return 1.
     */
    @Override
    public double calculateFirstDeriv(double input) {
        return 1;
    }

}
