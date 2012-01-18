/**
 * A generic node class to be used for neural networks.
 * 
 * @author Chao Zhang
 */

package com.network.node;

import com.function.Function;

public class Node {
    // This node's activation function; may be a continuous function.
    private final Function fun;
    // The last input through feed forwarding into this node.
    private double input;
    // The last output from this node.
    private double output;
    // The last error calculated from the last output.
    private double error;

    public Node(Function fun) {
        this.fun = fun;

        input = 0;
        output = 0;
        error = 0;
    }

    /**
     * Returns the function set for this node.
     * 
     * @return The function for this node.
     */
    public final Function getFun() {
        return fun;
    }

    /**
     * Returns the last input from feed forwarding for this node.
     * 
     * @return The last input for this node.
     */
    public final double getInput() {
        return input;
    }

    /**
     * Returns the last calculated output for this node.
     * 
     * @return The last output for this node.
     */
    public final double getOutput() {
        return output;
    }

    /**
     * Returns the last calculated error for this node.
     * 
     * @return The last error for this node.
     */
    public final double getError() {
        return error;
    }

    /**
     * Calculates the output of this node given an input using the function set
     * during initialization. Utilizes <code>Function</code>'s
     * <code>calculate</code> method.
     * 
     * @param input
     *            The input to the function.
     * @return The output from the function.
     */
    public double calculate(double input) {
        this.input = input;
        this.output = fun.calculate(input);
        return output;
    }

    /**
     * Calculates the error of this node using the last output. May require
     * auxiliary values to help calculate this error. If no auxiliary value is
     * needed, set <code>err</code> equal to 1.
     * 
     * This implementation will return 0 if <code>err</code> is set equal to 0.
     * 
     * @param err
     *            The auxiliary value, if needed, to calculate the error of this
     *            node. Should be set to 1 if such a value is not necessary.
     * @return The error for this node given its last output.
     */
    public double calculateError(double err) {
        error = fun.calculateFirstDeriv(output) * err;
        return error;
    }

}