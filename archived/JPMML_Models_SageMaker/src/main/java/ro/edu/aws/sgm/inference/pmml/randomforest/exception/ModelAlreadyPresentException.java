
package ro.edu.aws.sgm.inference.pmml.randomforest.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.CONFLICT)
public class ModelAlreadyPresentException extends RuntimeException{
    
    public ModelAlreadyPresentException(String exception){
        super(exception);
    }
}