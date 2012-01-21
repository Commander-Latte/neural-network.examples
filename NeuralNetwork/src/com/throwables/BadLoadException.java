package com.throwables;

public class BadLoadException extends Exception {

    /**
     * 
     */
    private static final long serialVersionUID = -6438469213887227832L;

    public BadLoadException() {
        super();
    }

    public BadLoadException(String message) {
        super(message);
    }
    
    public BadLoadException(String message, Throwable cause) {
        super(message, cause);
    }
    
    public BadLoadException(Throwable cause) {
        super(cause);
    }

}
