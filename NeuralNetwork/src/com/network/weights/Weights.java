/**
 * The weights to be used between nodes for the neural network.
 * 
 * @author Chao Zhang
 */

package com.network.weights;

public class Weights {
    // Number of nodes.
    private final int firstLayerSize;
    // Number of weights per node.
    private final int secondLayerSize;
    // Double array representation of the weights for each node.
    private double[][] weights;

    public Weights(int firstLayerSize, int secondLayerSize) {
        this.firstLayerSize = firstLayerSize;
        this.secondLayerSize = secondLayerSize;

        weights = new double[firstLayerSize][secondLayerSize];

        for (int x = 0; x < firstLayerSize; x++) {
            for (int y = 0; y < secondLayerSize; y++) {
                weights[x][y] = Math.random();
            }
        }
    }

    /**
     * Returns the weighted output for a node given the node's output, its
     * position in the first layer, and the receiving node's position in the
     * second layer.
     * 
     * This implementation returns the following:
     * 
     * weighted output = output * corresponding weight
     * 
     * @param output
     *            The output to be weighted.
     * @param firstLayerPos
     *            The output node's position in the first layer.
     * @param secondLayerPos
     *            The receiving node's position in the second layer.
     * @return The weighted output.
     * @throws IllegalArgumentException
     *             If either the output node or the receiving node's given
     *             positions lie outside of the bounds of what is valid for this
     *             weight object.
     */
    public double calculateWeightedOutput(double output, int firstLayerPos,
            int secondLayerPos) throws IllegalArgumentException {
        if (firstLayerPos >= firstLayerSize || firstLayerPos < 0) {
            throw new IllegalArgumentException(
                    "Invalid input node. Cannot calculate weighted output.");
        }
        if (secondLayerPos >= secondLayerSize || secondLayerPos < 0) {
            throw new IllegalArgumentException(
                    "Invalid input weight. Cannot calculate weighted output.");
        }

        return output * weights[firstLayerPos][secondLayerPos];
    }

    /**
     * Updates the weights for a node given the node's position in the first
     * layer, the position of the node in the second layer corresponding to the
     * weight to be modified, the error of this node in the second layer, the
     * momentum for learning, and the output of the node in the first layer.
     * 
     * This implementation is used for back propagation in the neural network.
     * It calculates the following:
     * 
     * new weight = old weight + momentum * error * first layer node's output
     * 
     * @param firstLayerPos
     *            The node's position in the first layer.
     * @param secondLayerPos
     *            The node in the second layer corresponding to the weight to be
     *            updated.
     * @param error
     *            The second layer node's error.
     * @param momentum
     *            The momentum for learning.
     * @param output
     *            The output of the node in the first layer.
     * @return The newly updated weight.
     * @throws IllegalArgumentException
     *             If either the output node or the receiving node's given
     *             positions lie outside of the bounds of what is valid for this
     *             weight object.
     */
    public double updateWeights(int firstLayerPos, int secondLayerPos,
            double error, double momentum, double output)
            throws IllegalArgumentException {
        if (firstLayerPos >= firstLayerSize || firstLayerPos < 0) {
            throw new IllegalArgumentException(
                    "Invalid input node. Cannot update weight.");
        }
        if (secondLayerPos >= secondLayerSize || secondLayerPos < 0) {
            throw new IllegalArgumentException(
                    "Invalid input weight. Cannot update weight.");
        }

        return weights[firstLayerPos][secondLayerPos] = weights[firstLayerPos][secondLayerPos]
                + momentum * error * output;
    }

    /**
     * Returns the weight given the two nodes the weight connects.
     * 
     * @param firstLayerNode
     *            The node in the first layer from which the weight extends.
     * @param secondLayerNode
     *            The node in the second layer to which the weight extends.
     * @return The weight connecting the two nodes.
     * @throws IllegalArgumentException
     *             If either the output node or the receiving node's given
     *             positions lie outside of the bounds of what is valid for this
     *             weight object.
     */
    public final double getWeight(int firstLayerNode, int secondLayerNode)
            throws IllegalArgumentException {
        if (firstLayerNode >= firstLayerSize || firstLayerNode < 0) {
            throw new IllegalArgumentException(
                    "Invalid input node. Cannot get weight.");
        }
        if (secondLayerNode >= secondLayerSize || secondLayerNode < 0) {
            throw new IllegalArgumentException(
                    "Invalid input weight. Cannot get weight.");
        }

        return weights[firstLayerNode][secondLayerNode];
    }

    /**
     * Returns the double array representation of the weights between two layers
     * of the neural network. Note that the returned weights are a copy of
     * actual weights to prevent tampering with actual values.
     * 
     * @return A copy of the weights between two layers in the neural network in
     *         double array form.
     */
    public final double[][] getWeights() {
        double[][] copy = new double[firstLayerSize][secondLayerSize];
        for (int x = 0; x < firstLayerSize; x++) {
            for (int y = 0; y < secondLayerSize; y++) {
                copy[x][y] = weights[x][y];
            }
        }
        return copy;
    }
    
    public final void setWeights(double[][] weights) {
        this.weights = weights;
    }
}
