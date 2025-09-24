package ro.edu.aws.sgm.inference.pmml.randomforest.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.INSUFFICIENT_STORAGE)
public class InsufficientMemoryException extends RuntimeException {

    public InsufficientMemoryException(String exception){
        super(exception);
    }
}
