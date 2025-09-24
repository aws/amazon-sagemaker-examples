package ro.edu.aws.sgm.inference.pmml.randomforest.handler;

import java.util.List;

import org.springframework.stereotype.Component;

import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.Features;

public interface InferenceHandlerInf {
    
    public String predict(List <Features> data, Object model);
}
