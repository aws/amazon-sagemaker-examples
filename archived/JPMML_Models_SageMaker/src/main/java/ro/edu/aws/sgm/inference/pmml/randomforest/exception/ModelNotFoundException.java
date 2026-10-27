package ro.edu.aws.sgm.inference.pmml.randomforest.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND)
public class ModelNotFoundException extends RuntimeException{
    
    public ModelNotFoundException(String exception){
        super(exception);
    }
}
