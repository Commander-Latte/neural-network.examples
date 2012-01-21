package com.readerwriter;

import java.io.IOException;
import java.util.Scanner;

import com.readerwriter.ReadWriteTextFile;
import com.throwables.BadLoadException;

public class Memory {

    private Memory() {
    }

    public final static boolean save(int[] layerCounts, double[] weights)
            throws IOException {
        return save("Output.txt", layerCounts, weights);
    }

    public final static boolean save(String fileName, int[] layerCounts,
            double[] weights) throws IOException {

        StringBuilder text = new StringBuilder();

        String NL = System.getProperty("line.separator");

        text.append("N" + NL);
        text.append(layerCounts.length + NL);

        text.append("L" + NL);
        for (int x = 0; x < layerCounts.length; x++) {
            text.append(layerCounts[x] + NL);
        }

        text.append("N" + NL);
        text.append(weights.length + NL);

        text.append("W" + NL);
        for (int x = 0; x < weights.length; x++) {
            text.append(weights[x] + NL);
        }

        return ReadWriteTextFile.write(fileName, text.toString());
    }

    public final static double[] load(String fileName, int[] layerCounts)
            throws IOException, BadLoadException {
        int numLayers = 0;
        int numWeights = 0;
        double[] weights = null;

        String storedText = ReadWriteTextFile.read(fileName);

        Scanner scan = new Scanner(storedText);

        consumeLetter(scan, "N");

        if (scan.hasNextInt()) {
            numLayers = scan.nextInt();
            if (numLayers != layerCounts.length) {
                throw new BadLoadException(
                        "Cannot use input file to load weights. Number of layers given in input file does not match that of the neural network.");
            }
        }
        else {
            throw new BadLoadException(
                    "Cannot use input file to load weights. Cannot read in number of layers.");
        }

        consumeLetter(scan, "L");

        int[] temp = new int[numLayers];

        for (int x = 0; x < numLayers; x++) {
            if (scan.hasNextInt()) {
                temp[x] = scan.nextInt();
            }
            else {
                throw new BadLoadException(
                        "Cannot use input file to load weights. Number of layers given in input file does not match that of the neural network.");
            }
        }

        for (int x = 0; x < numLayers; x++) {
            if (temp[x] != layerCounts[x]) {
                throw new BadLoadException(
                        "Cannot use input file to load weights. The number of nodes in layer "
                                + (x + 1)
                                + " given in the input file does not equal the number of nodes found in the neural network.");
            }
        }

        consumeLetter(scan, "N");

        if (scan.hasNextInt()) {
            numWeights = scan.nextInt();
            weights = new double[numWeights];
        }
        else {
            throw new BadLoadException(
                    "Cannot use input file to load weights. Cannot read in number of weights.");
        }

        consumeLetter(scan, "W");

        for (int x = 0; x < numWeights; x++) {
            if (scan.hasNextDouble()) {
                weights[x] = scan.nextDouble();
            }
            else {
                throw new BadLoadException(
                        "Cannot use input file to load weights. Not enough weights provided.");
            }
        }

        return weights;
    }

    private static void consumeLetter(Scanner scan, String letter)
            throws BadLoadException {
        if (scan.hasNext(letter)) {
            scan.next();
        }
        else {
            throw new BadLoadException(
                    "Cannot use input file to load weights. Bad input file format.");
        }
    }

//    public static void main(String[] args) {
//        String fileName = "Out.txt";
//        int[] layerCounts = { 2, 3, 4 };
//        double[] weights = { 5, 6, 7, 8, 9, 10 };
//
//        double[] out = null;
//
//        try {
//            // System.out.println(Memory.save(fileName, layerCounts, weights));
//            out = Memory.load(fileName, layerCounts);
//            for (double d : out)
//                System.out.println(d);
//        }
//        catch (IOException e) {
//            e.printStackTrace();
//            System.exit(1);
//        }
//        catch (BadLoadException b) {
//            b.printStackTrace();
//            System.exit(1);
//        }
//    }

}
