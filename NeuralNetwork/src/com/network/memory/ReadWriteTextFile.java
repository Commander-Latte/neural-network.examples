package com.network.memory;

import java.io.*;
import java.util.Scanner;

/**
 * Read and write a file using an explicit encoding. Removing the encoding from
 * this code will simply cause the system's default encoding to be used instead.
 * 
 * Modeled from code found on:
 * http://www.javapractices.com/topic/TopicAction.do?Id=42
 * 
 * @author Chao Zhang
 */
public final class ReadWriteTextFile {

    private static final boolean LOG = false;

    private ReadWriteTextFile() {
    }

    static boolean write(String fileName, String text)
            throws IOException {
        boolean successful = false;
        log("Writing to file named " + fileName + ".");
        Writer out = new OutputStreamWriter(new FileOutputStream(fileName));
        try {
            out.write(text);
            successful = true;
        }
        finally {
            out.close();
        }
        return successful;
    }

    static String read(String fileName) throws IOException, FileNotFoundException {
        log("Reading from file " + fileName + ".");

        StringBuilder outputText = new StringBuilder();

        String NL = System.getProperty("line.separator");

        Scanner scanner = new Scanner(new FileInputStream(fileName));
        try {
            while (scanner.hasNextLine()) {
                outputText.append(scanner.nextLine() + NL);
            }
        }
        finally {
            scanner.close();
        }
        log("Text read in: " + outputText);
        
        return outputText.toString();
    }

    private static void log(String aMessage) {
        if (LOG) {
            System.out.println(aMessage);
        }
    }
}
